"""Evaluate a selected OEM split and save metrics with checkpoint/configuration metadata."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import OpenEarthMapDataset  # noqa: E402
from src.metrics import evaluate_model  # noqa: E402
from src.checkpoints import load_model, checkpoint_metadata, file_sha256  # noqa: E402

IMAGE_SIZE = 512
NUM_CLASSES = 9
IGNORE_INDEX = 255


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a DualDecoder_SAM2UNet checkpoint on an OEM split')
    p.add_argument('--data-root', required=True, help='OpenEarthMap_wo_xBD root directory')
    p.add_argument('--split-txt', required=True, help='Path to the split file to evaluate (e.g. test.txt)')
    p.add_argument('--sam-checkpoint', required=True, help='SAM2 backbone checkpoint (sam2_hiera_tiny.pt)')
    p.add_argument('--weights', required=True, help='Fine-tuned model checkpoint (.pth)')
    p.add_argument('--variant', default='tiny', choices=['tiny', 'small', 'base', 'large'])
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--output', required=True, help='Evaluation JSON path')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f'Device: {device}')

    model = load_model(args.weights, args.sam_checkpoint, args.variant, device)
    print(f'Loaded checkpoint: {args.weights}')

    split_files = {'train': args.split_txt, 'val': args.split_txt, 'test': args.split_txt}
    dataset = OpenEarthMapDataset(Path(args.data_root), split=args.split, size=IMAGE_SIZE,
                                  mode='val', split_files=split_files)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f'Evaluating {len(dataset)} tiles...')

    metrics = evaluate_model(model, loader, device, num_classes=NUM_CLASSES,
                             ignore_index=IGNORE_INDEX, semantic_only=True)

    payload = {
        'metrics': metrics, 'checkpoint': checkpoint_metadata(args.weights, model),
        'configuration': vars(args), 'split': args.split,
        'split_sha256': file_sha256(args.split_txt),
        'sam_checkpoint_sha256': file_sha256(args.sam_checkpoint),
        'metric_settings': {'num_classes': 9, 'ignore_index': 255,
                            'class_zero_included': True, 'absent_classes': 'zero',
                            'boundary_dilation_ratio': .02, 'road_class': 4,
                            'averaging': 'macro across all nine classes'},
        'inference_path': 'semantic_only', 'image_size': IMAGE_SIZE,
    }
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

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