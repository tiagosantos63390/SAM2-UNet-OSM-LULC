"""Run inference with a trained DualDecoder_SAM2UNet on a single RGB image and
save the predicted land cover map as an indexed PNG using the 9-class
OpenEarthMap colour palette.

Example:
    python scripts/predict.py \
        --image /path/to/image.tif \
        --sam-checkpoint /path/to/sam2_hiera_tiny.pt \
        --weights /path/to/roads_bldg_L0p25_seed42_final.pth \
        --output prediction.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model import DualDecoder_SAM2UNet  # noqa: E402

IMAGE_SIZE = 512
NUM_CLASSES = 9

CLASS_LABELS = [
    'unclassified', 'bareland', 'rangeland', 'developed_space', 'road',
    'tree', 'water', 'agriculture', 'building',
]
CLASS_COLORS = [
    (0, 0, 0),         # 0: unclassified
    (128, 0, 0),       # 1: bareland
    (0, 255, 36),      # 2: rangeland
    (148, 148, 148),   # 3: developed_space
    (255, 255, 255),   # 4: road
    (34, 97, 38),      # 5: tree
    (0, 69, 255),      # 6: water
    (75, 181, 73),     # 7: agriculture
    (222, 31, 7),      # 8: building
]
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def parse_args():
    p = argparse.ArgumentParser(description='Predict a land cover map for a single RGB image')
    p.add_argument('--image', required=True, help='Input RGB image (any format PIL/OpenCV can open)')
    p.add_argument('--sam-checkpoint', required=True, help='SAM2 backbone checkpoint (sam2_hiera_tiny.pt)')
    p.add_argument('--weights', required=True, help='Fine-tuned model checkpoint (.pth)')
    p.add_argument('--output', default=None, help='Output PNG path (default: <image stem>_pred.png)')
    p.add_argument('--variant', default='tiny', choices=['tiny', 'small', 'base', 'large'])
    p.add_argument('--size', type=int, default=IMAGE_SIZE, help='Model input size (default: 512)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_weights(model, weights_path):
    state = torch.load(weights_path, map_location='cpu')
    if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
        state = state.get('model_state_dict', state.get('state_dict'))
    if isinstance(state, dict) and len(state) > 0 and next(iter(state)).startswith('module.'):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint/state-dict mismatch - missing: {len(missing)}, unexpected: {len(unexpected)}')


def palette_png(mask, output_path):
    palette = []
    for color in CLASS_COLORS:
        palette.extend(color)
    palette.extend([0] * (768 - len(palette)))
    img = Image.fromarray(mask.astype(np.uint8), mode='P')
    img.putpalette(palette)
    img.save(output_path)


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = DualDecoder_SAM2UNet(checkpoint_path=args.sam_checkpoint, variant=args.variant,
                                 num_classes=NUM_CLASSES).to(device)
    load_weights(model, args.weights)
    model.eval()
    print(f'Loaded checkpoint: {args.weights}')

    image = Image.open(args.image).convert('RGB')
    image = transforms.Resize((args.size, args.size), transforms.InterpolationMode.BILINEAR)(image)
    x = transforms.ToTensor()(image)
    x = NORMALIZE(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_oem = model(x)[0]
        pred = torch.argmax(logits_oem, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    output = args.output or str(Path(args.image).with_suffix('')) + '_pred.png'
    palette_png(pred, output)
    print('-' * 40)
    for cls, count in zip(*np.unique(pred, return_counts=True)):
        print(f'{CLASS_LABELS[int(cls)]:<18}: {int(count)} px')
    print('-' * 40)
    print(f'Saved prediction to {output}')


if __name__ == '__main__':
    main()