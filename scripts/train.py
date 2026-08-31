"""Train the paper models: baseline + OSM auxiliary supervision runs over
several seeds. The training logic is taken line-for-line from the final
training notebook (notebooks/train_paper_models_seeds42-44.ipynb); only the
Kaggle paths and the seed list are exposed as command-line arguments.

Example:
    python scripts/train.py \
        --data-root /path/to/OpenEarthMap_wo_xBD \
        --train-txt train.txt --val-txt val.txt --test-txt test.txt \
        --osm-roads-dir /path/to/osm/roads/binary_masks \
        --osm-buildings-dir /path/to/osm/buildings/binary_masks \
        --sam-checkpoints-dir /path/to/sam-checkpoints \
        --seeds 42 43 44
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import OpenEarthMapDataset  # noqa: E402
from src.metrics import evaluate_model  # noqa: E402
from src.model import DualDecoder_SAM2UNet  # noqa: E402

# ==============================================================================
# Training configuration (as in the notebook)
# ==============================================================================
EPOCHS = 150
EVAL_EVERY = 1
IMAGE_SIZE = 512
NUM_CLASSES = 9
BATCH_SIZE_DEFAULT = 4
LR = 1e-3
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 2
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 0.0001

CKPT_PATTERNS = {'tiny': 'sam2_hiera_tiny.pt'}


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline and OSM-auxiliary SAM2-UNet models")
    p.add_argument('--data-root', required=True, help='OpenEarthMap_wo_xBD root directory')
    p.add_argument('--train-txt', required=True)
    p.add_argument('--val-txt', required=True)
    p.add_argument('--test-txt', required=True)
    p.add_argument('--osm-roads-dir', default=None, help='Rasterised OSM road masks')
    p.add_argument('--osm-buildings-dir', default=None, help='Rasterised OSM building masks')
    p.add_argument('--sam-checkpoints-dir', required=True, help='Directory with SAM2 checkpoints')
    p.add_argument('--sam-checkpoint', default='sam2_hiera_tiny.pt', help='Checkpoint filename inside --sam-checkpoints-dir')
    p.add_argument('--seeds', type=int, nargs='+', default=[42], help='Random seeds to run (one Kaggle session per seed fits the GPU budget)')
    p.add_argument('--work-dir', default='./sam2unet_augmented_osm_run')
    return p.parse_args()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, metric_name='miou'):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.counter = 0
        self.best_metric = None
        self.best_epoch = None

    def __call__(self, current_metric, epoch):
        if self.best_metric is None or current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _find_ckpt(root: Path, pattern):
    direct = root / pattern
    if direct.exists(): return direct
    nested = list(root.rglob(pattern))
    return nested[0] if nested else None


def _distance_transform_weights(mask, max_distance):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)
    valid = mask_np != 255
    if not np.any(valid):
        return np.ones_like(mask_np, dtype=np.float32)
    binary = (mask_np > 0).astype(np.uint8)
    dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    max_distance = float(max_distance)
    weights = np.where(dist <= max_distance, (max_distance - dist) / max(max_distance, 1e-6), 0.0)
    weights[~valid] = 0.0
    return weights.astype(np.float32)


def main():
    args = parse_args()

    SEEDS_TO_RUN = args.seeds
    MODEL_VARIANTS = ['tiny']

    WORK_DIR = Path(args.work_dir)
    OUTPUT_DIR = WORK_DIR / 'outputs'
    HISTORY_DIR = WORK_DIR / 'history'
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ckpt_root = Path(args.sam_checkpoints_dir)
    variant_ckpts = {v: _find_ckpt(ckpt_root, CKPT_PATTERNS[v]) for v in MODEL_VARIANTS}

    DATA_ROOT = Path(args.data_root)
    OSM_DATASET_PATH = Path(args.osm_roads_dir) if args.osm_roads_dir else None
    OSM_BLDG_DATASET_PATH = Path(args.osm_buildings_dir) if args.osm_buildings_dir else None

    split_files = {'train': args.train_txt, 'val': args.val_txt, 'test': args.test_txt}

    history_all = {}

    run_plan = [
        {'name': 'baseline', 'label': 'Baseline (OEM Only)', 'lambda_osm': 0.0, 'lambda_bldg': 0.0, 'use_osm_data': False, 'use_bldg_data': False, 'dt_max_distance': 6.0},
        {'name': 'roads_only_L0p5', 'label': 'Roads Only (L=0.5)', 'lambda_osm': 0.5, 'lambda_bldg': 0.0, 'use_osm_data': True, 'use_bldg_data': False, 'dt_max_distance': 6.0},
        {'name': 'roads_bldg_L0p25', 'label': 'Roads+Bldg (L_bldg=0.25)', 'lambda_osm': 0.5, 'lambda_bldg': 0.25, 'use_osm_data': True, 'use_bldg_data': True, 'dt_max_distance': 6.0},
        {'name': 'roads_bldg_L0p5', 'label': 'Roads+Bldg (L_bldg=0.5)', 'lambda_osm': 0.5, 'lambda_bldg': 0.5, 'use_osm_data': True, 'use_bldg_data': True, 'dt_max_distance': 6.0},
        {'name': 'roads_bldg_L0p75', 'label': 'Roads+Bldg (L_bldg=0.75)', 'lambda_osm': 0.5, 'lambda_bldg': 0.75, 'use_osm_data': True, 'use_bldg_data': True, 'dt_max_distance': 6.0},
    ]

    for SEED in SEEDS_TO_RUN:
        print('\n' + '=' * 100)
        print(f'STARTING SEED: {SEED}')
        print('=' * 100)
        
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)

        for run in run_plan:
            variant = 'tiny'
            run_name = f"{run['name']}_seed{SEED}"
            
            run_osm_dir = OSM_DATASET_PATH if run['use_osm_data'] else None
            run_bldg_dir = OSM_BLDG_DATASET_PATH if run['use_bldg_data'] else None
            
            train_dataset = OpenEarthMapDataset(DATA_ROOT, osm_dir=run_osm_dir, bldg_dir=run_bldg_dir, split='train', size=IMAGE_SIZE, mode='train', seed=SEED, split_files=split_files)
            val_dataset = OpenEarthMapDataset(DATA_ROOT, osm_dir=run_osm_dir, bldg_dir=run_bldg_dir, split='val', size=IMAGE_SIZE, mode='val', seed=SEED, split_files=split_files)
            test_dataset = OpenEarthMapDataset(DATA_ROOT, osm_dir=run_osm_dir, bldg_dir=run_bldg_dir, split='test', size=IMAGE_SIZE, mode='val', seed=SEED, split_files=split_files)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

            print(f"\nTraining run: {run_name} | {run['label']}")
            
            model = DualDecoder_SAM2UNet(checkpoint_path=str(variant_ckpts[variant]), variant=variant, num_classes=NUM_CLASSES).to(device)

            criterion_oem = torch.nn.CrossEntropyLoss(ignore_index=255)
            criterion_osm = torch.nn.MSELoss()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-7)
            scaler = GradScaler('cuda')
            early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, metric_name='miou')

            run_dir = OUTPUT_DIR / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            history = []
            actual_epochs = 0
            run_best_model_state = None
            run_best_metrics = None

            for epoch in range(1, EPOCHS + 1):
                model.train()
                running_loss = 0.0

                for step, batch in enumerate(train_loader):
                    images = batch['image'].to(device, non_blocking=True)
                    masks_oem = batch['label'].to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    with autocast('cuda'):
                        logits_oem, logits_osm = model(images)
                        loss = criterion_oem(logits_oem, masks_oem)
                        
                        if run['use_osm_data'] and 'osm' in batch:
                            masks_osm = batch['osm'].to(device, non_blocking=True)
                            road_probs = torch.sigmoid(logits_osm[:, 0:1])
                            dt_targets = torch.from_numpy(
                                np.stack([_distance_transform_weights(mask, run['dt_max_distance']) for mask in masks_osm.detach().cpu().numpy()])
                            ).to(device=device, dtype=road_probs.dtype).unsqueeze(1)
                            loss = loss + (run['lambda_osm'] * criterion_osm(road_probs, dt_targets))
                            
                        if run['use_bldg_data'] and 'osm_bldg' in batch:
                            masks_bldg = batch['osm_bldg'].to(device, non_blocking=True)
                            bldg_probs = torch.sigmoid(logits_osm[:, 1:2])
                            dt_targets_bldg = torch.from_numpy(
                                np.stack([_distance_transform_weights(mask, run['dt_max_distance']) for mask in masks_bldg.detach().cpu().numpy()])
                            ).to(device=device, dtype=bldg_probs.dtype).unsqueeze(1)
                            loss = loss + (run['lambda_bldg'] * criterion_osm(bldg_probs, dt_targets_bldg))

                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += loss.item()
                    if step % 100 == 0:
                        print(f'[{run_name}] Epoch [{epoch}/{EPOCHS}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}')

                scheduler.step()
                avg_loss = running_loss / max(len(train_loader), 1)

                do_eval = (epoch % EVAL_EVERY == 0) or (epoch == EPOCHS)
                metrics = None
                if do_eval:
                    metrics = evaluate_model(model, val_loader, device, num_classes=NUM_CLASSES, ignore_index=255, is_dual=True)
                    print(f"[{run_name}] Epoch [{epoch}/{EPOCHS}] mIoU={metrics['miou']:.4f} B-mIoU={metrics['bmiou']:.4f} RoadIoU={metrics['road_iou']:.4f}")

                    if run_best_metrics is None or metrics['miou'] > run_best_metrics['miou']:
                        run_best_model_state = model.state_dict().copy()
                        run_best_metrics = metrics

                    if early_stopper(metrics['miou'], epoch):
                        print(f'[{run_name}] Early stopping at epoch {epoch}. Best mIoU={early_stopper.best_metric:.4f}')
                        actual_epochs = epoch
                        break

                actual_epochs = epoch
                history.append({
                    'epoch': epoch, 'miou': float(metrics['miou']) if metrics else None,
                    'bmiou': float(metrics['bmiou']) if metrics else None, 'road_iou': float(metrics['road_iou']) if metrics else None,
                })

            # Save Metrics
            model.load_state_dict(run_best_model_state)
            test_metrics = evaluate_model(model, test_loader, device, num_classes=NUM_CLASSES, ignore_index=255, is_dual=True)

            metrics_out = {
                'test': test_metrics, 'best_epoch': early_stopper.best_epoch if early_stopper.best_epoch is not None else actual_epochs,
                'best_miou': early_stopper.best_metric, 'seed': SEED
            }

            with open(HISTORY_DIR / f'{run_name}_metrics.json', 'w', encoding='utf-8') as f:
                json.dump(metrics_out, f, indent=2)

            history_all[run_name] = {
                'name': run_name, 
                'label': run['label'], 
                'seed': SEED, 
                'history': history, 
                'final_metrics': metrics_out
            }
            
            # Save model immediately to prevent data loss!
            torch.save(run_best_model_state, run_dir / f'{run_name}_final.pth')
            print(f'[{run_name}] Final model safely saved to: {run_dir / f"{run_name}_final.pth"}')
            
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    print('\nTraining complete for all selected seeds.')


if __name__ == '__main__':
    main()
