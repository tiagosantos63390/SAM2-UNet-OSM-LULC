"""Evaluate a trained DualDecoder_SAM2UNet checkpoint on the OpenEarthMap test
split and print the same metrics used in the paper (mIoU, F1, Boundary mIoU,
per-class IoU, road IoU).

Example:
    python scripts/evaluate.py \
        --data-root /path/to/OpenEarthMap_wo_xBD \
        --split-txt test.txt \
        --sam-checkpoint /path/to/sam2_hiera_tiny.pt \
        --weights /path/to/roads_bldg_L0p25_seed42_final.pth \
        --variant tiny
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import OpenEarthMapDataset  # noqa: E402
from src.metrics import evaluate_model  # noqa: E402
from src.model import DualDecoder_SAM2UNet  # noqa: E402

IMAGE_SIZE = 512
NUM_CLASSES = 9
IGNORE_INDEX = 255


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a DualDecoder_SAM2UNet checkpoint on the test split')
    p.add_argument('--data-root', required=True, help='OpenEarthMap_wo_xBD root directory')
    p.add_argument('--split-txt', required=True, help='Path to the split file to evaluate (e.g. test.txt)')
    p.add_argument('--sam-checkpoint', required=True, help='SAM2 backbone checkpoint (sam2_hiera_tiny.pt)')
    p.add_argument('--weights', required=True, help='Fine-tuned model checkpoint (.pth)')
    p.add_argument('--variant', default='tiny', choices=['tiny', 'small', 'base', 'large'])
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f'Device: {device}')

    model = DualDecoder_SAM2UNet(checkpoint_path=args.sam_checkpoint, variant=args.variant,
                                 num_classes=NUM_CLASSES).to(device)

    state = torch.load(args.weights, map_location='cpu')
    if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
        state = state.get('model_state_dict', state.get('state_dict'))
    if isinstance(state, dict) and len(state) > 0 and next(iter(state)).startswith('module.'):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint/state-dict mismatch - missing: {len(missing)}, unexpected: {len(unexpected)}')
    model.eval()
    print(f'Loaded checkpoint: {args.weights}')

    split_files = {'train': args.split_txt, 'val': args.split_txt, 'test': args.split_txt}
    dataset = OpenEarthMapDataset(Path(args.data_root), split='test', size=IMAGE_SIZE,
                                  mode='val', split_files=split_files)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f'Evaluating {len(dataset)} tiles...')

    metrics = evaluate_model(model, loader, device, num_classes=NUM_CLASSES,
                             ignore_index=IGNORE_INDEX, is_dual=True)

    labels = ['unclassified', 'bareland', 'rangeland', 'developed_space', 'road',
              'tree', 'water', 'agriculture', 'building']
    print('-' * 40)
    print(f"mIoU        : {metrics['miou']:.4f}")
    print(f"F1          : {metrics['f1']:.4f}")
    print(f"Precision   : {metrics['precision']:.4f}")
    print(f"Recall      : {metrics['recall']:.4f}")
    print(f"Boundary mIo: {metrics['bmiou']:.4f}")
    print(f"Road IoU    : {metrics['road_iou']:.4f}")
    print('-' * 40)
    for name, iou in zip(labels, metrics['per_class_iou']):
        print(f'{name:<18}: {iou:.4f}')


if __name__ == '__main__':
    main()