"""Train baseline and OSM auxiliary experiments over several seeds.

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

AUX_TASKS = (
    ('road', 'osm', 'osm_masks', 'lambda_osm', 'use_osm_data'),
    ('building', 'osm_bldg', 'bldg_masks', 'lambda_bldg', 'use_bldg_data'),
    ('water', 'osm_water', 'water_masks', 'lambda_water', 'use_water_data'),
    ('tree', 'osm_tree', 'tree_masks', 'lambda_tree', 'use_tree_data'),
)


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline and OSM-auxiliary SAM2-UNet models")
    p.add_argument('--data-root', required=True, help='OpenEarthMap_wo_xBD root directory')
    p.add_argument('--train-txt', required=True)
    p.add_argument('--val-txt', required=True)
    p.add_argument('--test-txt', required=True)
    p.add_argument('--osm-roads-dir', default=None, help='Rasterised OSM road masks')
    p.add_argument('--osm-buildings-dir', default=None, help='Rasterised OSM building masks')
    p.add_argument('--osm-water-dir', default=None, help='Rasterised OSM water masks')
    p.add_argument('--osm-trees-dir', default=None, help='Rasterised OSM tree masks')
    p.add_argument('--experiments', nargs='+', help='Run only these experiment names (default: existing five runs)')
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
    requested = Path(pattern)
    if requested.is_absolute():
        if requested.is_file():
            return requested.resolve()
        raise FileNotFoundError(f'Backbone checkpoint not found: {requested}')
    direct = root / requested
    if direct.is_file():
        return direct.resolve()
    nested = sorted(path for path in root.rglob(str(requested)) if path.is_file())
    if len(nested) > 1:
        raise ValueError(f'Multiple backbone checkpoints match {requested} under {root}: {nested}')
    if not nested:
        raise FileNotFoundError(f'Backbone checkpoint {requested} not found under {root}')
    return nested[0].resolve()


def _distance_transform_weights(mask, max_distance):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)
    valid = (mask_np == 0) | (mask_np == 1)
    binary = (mask_np == 1).astype(np.uint8)
    if not np.any(binary):
        return np.zeros_like(mask_np, dtype=np.float32)
    dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    max_distance = float(max_distance)
    weights = np.where(dist <= max_distance, (max_distance - dist) / max(max_distance, 1e-6), 0.0)
    weights[~valid] = 0.0
    return weights.astype(np.float32)


def auxiliary_loss(logits, masks, max_distance, valid=None):
    if valid is None:
        valid = (masks == 0) | (masks == 1)
    masks = masks.masked_fill(~valid, 255)
    targets = torch.from_numpy(np.stack([
        _distance_transform_weights(mask, max_distance)
        for mask in masks.detach().cpu().numpy()
    ])).to(device=logits.device, dtype=torch.float32)
    errors = (torch.sigmoid(logits[:, 0].float()) - targets).square()
    return (errors * valid).sum() / valid.sum().clamp_min(1)


def building_bce_dice_loss(logits, masks, valid):
    # v8: equal BCE/Dice weights, spatial Dice per sample, smooth=1e-6.
    logits = logits.float()
    targets = (masks == 1).float().unsqueeze(1)
    valid = valid.unsqueeze(1).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    bce_loss = (bce * valid).sum() / valid.sum().clamp_min(1)
    probs, gt = torch.sigmoid(logits) * valid, targets * valid
    inter = (probs * gt).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + gt.sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1e-6) / (union + 1e-6)
    usable = valid.sum(dim=(2, 3)) > 0
    dice_loss = (dice * usable).sum() / usable.sum().clamp_min(1)
    return .5 * bce_loss + .5 * dice_loss


def seed_run(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    dataset = torch.utils.data.get_worker_info().dataset
    for transform in dataset.transform.transforms:
        if hasattr(transform, 'aug') and hasattr(transform.aug, 'set_random_seed'):
            transform.aug.set_random_seed(seed)


def check_auxiliary_supervision(loader, run):
    enabled = [(key, paths) for _, key, paths, _, flag in AUX_TASKS if run.get(flag, False)]
    dataset = loader.dataset
    for key, paths in enabled:
        matched = sum(p is not None for p in getattr(dataset, paths))
        print(f'{key}: {matched}/{len(dataset)} masks matched')
        if not matched:
            raise ValueError(f'Enabled auxiliary task {key} has no usable training masks (no matches)')
    usable = {key: 0 for key, _ in enabled}
    foreground = dict(usable)
    checked = 0
    if enabled:
        for batch in loader:
            checked += len(batch['image'])
            for key, _ in enabled:
                usable[key] += int(batch[key + '_valid'].flatten(1).any(1).sum())
                foreground[key] += int(((batch[key] == 1) & batch[key + '_valid']).flatten(1).any(1).sum())
            if all(usable.values()) and all(foreground.values()):
                break
    for key, _ in enabled:
        print(f'{key}: {usable[key]}/{checked} inspected tiles usable, {foreground[key]}/{checked} with valid foreground')
        if not usable[key]:
            raise ValueError(f'Enabled auxiliary task {key} has no usable training masks')
        if not foreground[key]:
            raise ValueError(f'Enabled auxiliary task {key} has no valid foreground in {checked} inspected training tiles')


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
    variant_ckpts = {v: _find_ckpt(ckpt_root, args.sam_checkpoint) for v in MODEL_VARIANTS}

    DATA_ROOT = Path(args.data_root)
    OSM_DATASET_PATH = Path(args.osm_roads_dir) if args.osm_roads_dir else None
    OSM_BLDG_DATASET_PATH = Path(args.osm_buildings_dir) if args.osm_buildings_dir else None
    water_dir = Path(args.osm_water_dir) if args.osm_water_dir else None
    tree_dir = Path(args.osm_trees_dir) if args.osm_trees_dir else None

    split_files = {'train': args.train_txt, 'val': args.val_txt, 'test': args.test_txt}

    history_all = {}

    run_plan = [
        {'name': 'baseline', 'label': 'Baseline (OEM Only)', 'lambda_osm': 0.0, 'lambda_bldg': 0.0, 'use_osm_data': False, 'use_bldg_data': False, 'dt_max_distance': 6.0},
        {'name': 'roads_only_L0p5', 'label': 'Roads Only (L=0.5)', 'lambda_osm': 0.5, 'lambda_bldg': 0.0, 'use_osm_data': True, 'use_bldg_data': False, 'dt_max_distance': 6.0},
        {'name': 'roads_bldg_L0p25', 'label': 'Roads+Bldg (L_bldg=0.25)', 'lambda_osm': 0.5, 'lambda_bldg': 0.25, 'use_osm_data': True, 'use_bldg_data': True, 'dt_max_distance': 6.0},
        {'name': 'roads_bldg_L0p5', 'label': 'Roads+Bldg (L_bldg=0.5)', 'lambda_osm': 0.5, 'lambda_bldg': 0.5, 'use_osm_data': True, 'use_bldg_data': True, 'dt_max_distance': 6.0},
        {'name': 'roads_bldg_L0p75', 'label': 'Roads+Bldg (L_bldg=0.75)', 'lambda_osm': 0.5, 'lambda_bldg': 0.75, 'use_osm_data': True, 'use_bldg_data': True, 'dt_max_distance': 6.0},
    ]
    default_names = [run['name'] for run in run_plan]
    for weight, label in ((.25, '0p25'), (.5, '0p5')):
        run_plan.append({
            'name': f'bldg_only_L{label}_bce_dice', 'label': f'Buildings Only (L={weight}, BCE+Dice)',
            'lambda_osm': 0.0, 'lambda_bldg': weight, 'use_osm_data': False,
            'use_bldg_data': True, 'building_loss': 'bce_dice', 'dt_max_distance': 6.0,
        })
    for formulation in ('dt_mse', 'mixed'):
        run_plan.append({
            'name': f'four_class_L0p5_d6_{formulation}', 'label': f'Four-Class (L=0.5, {formulation})',
            'lambda_osm': .5, 'lambda_bldg': .5, 'lambda_water': .5, 'lambda_tree': .5,
            'use_osm_data': True, 'use_bldg_data': True, 'use_water_data': True, 'use_tree_data': True,
            'aux_channels': 4, 'building_loss': 'bce_dice' if formulation == 'mixed' else 'dt_mse',
            'dt_max_distance': 6.0,
        })
    available = {run['name']: run for run in run_plan}
    selected = args.experiments if args.experiments is not None else default_names
    unknown = set(selected) - available.keys()
    if unknown:
        raise ValueError(f'Unknown experiments: {sorted(unknown)}. Choose from: {list(available)}')
    run_plan = [available[name] for name in dict.fromkeys(selected)]

    for SEED in SEEDS_TO_RUN:
        print('\n' + '=' * 100)
        print(f'STARTING SEED: {SEED}')
        print('=' * 100)
        
        for run in run_plan:
            seed_run(SEED)
            variant = 'tiny'
            run_name = f"{run['name']}_seed{SEED}"
            
            run_osm_dir = OSM_DATASET_PATH if run['use_osm_data'] else None
            run_bldg_dir = OSM_BLDG_DATASET_PATH if run['use_bldg_data'] else None
            extra_dirs = {
                'water_dir': water_dir if run.get('use_water_data', False) else None,
                'tree_dir': tree_dir if run.get('use_tree_data', False) else None,
            }
            aux_channels = run.get('aux_channels', 2)
            auxiliary = {
                name: {'channel': channel, 'sample_key': key, 'weight': run.get(coefficient, 0.0),
                       'loss': run.get('building_loss', 'dt_mse') if name == 'building' else 'dt_mse'}
                for channel, (name, key, _, coefficient, _) in enumerate(AUX_TASKS)
                if channel < aux_channels
            }
            
            train_dataset = OpenEarthMapDataset(DATA_ROOT, osm_dir=run_osm_dir, bldg_dir=run_bldg_dir, split='train', size=IMAGE_SIZE, mode='train', seed=SEED, split_files=split_files, **extra_dirs)
            val_dataset = OpenEarthMapDataset(DATA_ROOT, osm_dir=run_osm_dir, bldg_dir=run_bldg_dir, split='val', size=IMAGE_SIZE, mode='val', seed=SEED, split_files=split_files, **extra_dirs)
            test_dataset = OpenEarthMapDataset(DATA_ROOT, osm_dir=run_osm_dir, bldg_dir=run_bldg_dir, split='test', size=IMAGE_SIZE, mode='val', seed=SEED, split_files=split_files, **extra_dirs)

            generator = torch.Generator().manual_seed(SEED)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, generator=generator, worker_init_fn=seed_worker)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

            print(f"\nTraining run: {run_name} | {run['label']}")
            check_auxiliary_supervision(train_loader, run)
            # Reset shuffle and augmentation seeds after the startup check.
            seed_run(SEED)
            generator.manual_seed(SEED)
            for transform in train_dataset.transform.transforms:
                if hasattr(transform, 'aug') and hasattr(transform.aug, 'set_random_seed'):
                    transform.aug.set_random_seed(SEED)
            
            model = DualDecoder_SAM2UNet(checkpoint_path=str(variant_ckpts[variant]), variant=variant, num_classes=NUM_CLASSES, aux_channels=aux_channels).to(device)

            criterion_oem = torch.nn.CrossEntropyLoss(ignore_index=255)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-7)
            scaler = GradScaler('cuda', enabled=device.type == 'cuda')
            early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, metric_name='miou')

            run_dir = OUTPUT_DIR / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / f'{run_name}_final.pth'
            if (checkpoint.exists() or (HISTORY_DIR / f'{run_name}_metrics.json').exists()
                    or (HISTORY_DIR / f'{run_name}_history.json').exists()):
                raise FileExistsError(f'Run outputs already exist for {run_name}; choose a new --work-dir')

            history = []
            actual_epochs = 0
            best_epoch = None
            run_best_metrics = None

            for epoch in range(1, EPOCHS + 1):
                model.train()
                running_loss = 0.0

                for step, batch in enumerate(train_loader):
                    images = batch['image'].to(device, non_blocking=True)
                    masks_oem = batch['label'].to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    with autocast('cuda', enabled=device.type == 'cuda'):
                        logits_oem, logits_osm = model(images)
                        loss = criterion_oem(logits_oem, masks_oem)
                        
                        for task in auxiliary.values():
                            if task['weight'] == 0:
                                continue
                            key, channel = task['sample_key'], task['channel']
                            masks = batch[key].to(device, non_blocking=True)
                            valid = batch[key + '_valid'].to(device, non_blocking=True)
                            logits = logits_osm[:, channel:channel + 1]
                            if task['loss'] == 'bce_dice':
                                task_loss = building_bce_dice_loss(logits, masks, valid)
                            else:
                                task_loss = auxiliary_loss(logits, masks, run['dt_max_distance'], valid)
                            loss = loss + task['weight'] * task_loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
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
                        torch.save(model.state_dict(), checkpoint)
                        run_best_metrics = metrics
                        best_epoch = epoch

                    should_stop = early_stopper(metrics['miou'], epoch)
                    if should_stop:
                        print(f'[{run_name}] Early stopping at epoch {epoch}. Best mIoU={early_stopper.best_metric:.4f}')

                actual_epochs = epoch
                history.append({
                    'epoch': epoch, 'miou': float(metrics['miou']) if metrics else None,
                    'bmiou': float(metrics['bmiou']) if metrics else None, 'road_iou': float(metrics['road_iou']) if metrics else None,
                })
                if do_eval and should_stop:
                    break

            # Save Metrics
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            test_metrics = evaluate_model(model, test_loader, device, num_classes=NUM_CLASSES, ignore_index=255, is_dual=True)

            metrics_out = {
                'test': test_metrics, 'best_epoch': best_epoch, 'stopping_epoch': actual_epochs,
                'best_miou': run_best_metrics['miou'], 'seed': SEED,
                'configuration': {**vars(args), **run, 'variant': variant, 'epochs': EPOCHS,
                                  'aux_channels': aux_channels, 'auxiliary': auxiliary,
                                  'image_size': IMAGE_SIZE, 'num_classes': NUM_CLASSES,
                                  'batch_size': BATCH_SIZE_DEFAULT, 'lr': LR, 'weight_decay': WEIGHT_DECAY,
                                  'num_workers': NUM_WORKERS, 'eval_every': EVAL_EVERY,
                                  'patience': EARLY_STOP_PATIENCE, 'min_delta': EARLY_STOP_MIN_DELTA,
                                  'eta_min': 1e-7, 'gradient_clip_norm': 1.0,
                                  'sam_checkpoint': str(variant_ckpts[variant]),
                                  'mask_encoding': '0/1; other values invalid'},
                'checkpoint': str(checkpoint),
            }

            with open(HISTORY_DIR / f'{run_name}_metrics.json', 'w', encoding='utf-8') as f:
                json.dump(metrics_out, f, indent=2)
            with open(HISTORY_DIR / f'{run_name}_history.json', 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

            history_all[run_name] = {
                'name': run_name, 
                'label': run['label'], 
                'seed': SEED, 
                'history': history, 
                'final_metrics': metrics_out
            }
            
            print(f'[{run_name}] Final model safely saved to: {run_dir / f"{run_name}_final.pth"}')
            
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    print('\nTraining complete for all selected seeds.')


if __name__ == '__main__':
    main()
