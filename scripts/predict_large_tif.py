"""Run sliding-window inference with a trained DualDecoder_SAM2UNet on a large
geoTIFF (e.g. a UAV / drone orthomosaic) and stitch the predictions into a
single indexed PNG using the 9-class OpenEarthMap colour palette.

Adapted from the UAV inference script used for the paper's drone example:
- the released DualDecoder_SAM2UNet (@512, ImageNet-normalised input, OEM head)
  replaces the older single-decoder SAM2UNet,
- the tile size is derived from the image's ground sampling distance (GSD) so a
  fixed physical footprint (~target_gsd m/px) is always shown to the model.

Example:
    python scripts/predict_large_tif.py \
        --tif /path/to/uav_orthomosaic.tif \
        --sam-checkpoint /path/to/sam2_hiera_tiny.pt \
        --weights /path/to/roads_bldg_L0p25_seed42_final.pth \
        --output uav_prediction
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model import DualDecoder_SAM2UNet  # noqa: E402

MODEL_INPUT_SIZE = 512
NUM_CLASSES = 9
TARGET_GSD = 0.5
OVERLAP_RATIO = 0.2

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

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


def load_weights(model, weights_path):
    state = torch.load(weights_path, map_location='cpu')
    if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
        state = state.get('model_state_dict', state.get('state_dict'))
    if isinstance(state, dict) and len(state) > 0 and next(iter(state)).startswith('module.'):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint/state-dict mismatch - missing: {len(missing)}, unexpected: {len(unexpected)}')


def get_tile_starts_with_stride(dimension_length, tile_size, stride):
    """Tile starts with overlap (stride < tile_size) and full coverage."""
    stride = max(1, min(stride, tile_size))
    starts = list(range(0, dimension_length, stride))
    if starts[-1] + tile_size < dimension_length:
        starts.append(max(0, dimension_length - tile_size))
    return sorted(list(set(starts)))


def mask_to_rgb(mask):
    """Map a 2D class mask to RGB using a lookup table (chunked to save memory)."""
    lut = np.array(CLASS_COLORS, dtype=np.uint8)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    chunk = 2048
    for i in range(0, mask.shape[0], chunk):
        rgb[i:i + chunk] = lut[mask[i:i + chunk]]
    return rgb


def save_palette_png(mask_full, output_path):
    palette = []
    for color in CLASS_COLORS:
        palette.extend(color)
    palette.extend([0] * (768 - len(palette)))
    img = Image.fromarray(mask_full, mode='P')
    img.putpalette(palette)
    img.save(output_path)


def predict_mask_on_large_tif(tif_path, weights_path, sam_checkpoint, output_name, variant='tiny',
                              model_input_size=MODEL_INPUT_SIZE, user_resolution=None,
                              target_gsd=TARGET_GSD, overlap_ratio=OVERLAP_RATIO,
                              device=None, preview=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DualDecoder_SAM2UNet(checkpoint_path=sam_checkpoint, variant=variant,
                                 num_classes=NUM_CLASSES).to(device)
    load_weights(model, weights_path)
    model.eval()
    print(f'Loaded model ({variant}) and checkpoint: {weights_path}')

    with rasterio.open(tif_path) as src:
        orig_h, orig_w = src.height, src.width
        img_res = user_resolution if user_resolution else src.res[0]

        if img_res < 0.001:
            print(f'WARNING: Detected resolution {img_res:.6f} seems to be in degrees. Defaulting to 8192 crop.')
            tile_size = 8192
        elif 0.20 <= img_res <= 0.60:
            tile_size = int(model_input_size * (target_gsd / img_res))
            tile_size = max(model_input_size, round(tile_size / 1024) * 1024)
            print(f'Detected Image Resolution (GSD): {img_res:.4f} meters/pixel')
            print(f'Using OEM-preserving crop size: {tile_size}x{tile_size} pixels')
        else:
            tile_size = int(model_input_size * (target_gsd / img_res))
            tile_size = max(1024, round(tile_size / 1024) * 1024)
            print(f'Detected Image Resolution (GSD): {img_res:.4f} meters/pixel')
            print(f'Calculated Optimal Crop Size: {tile_size}x{tile_size} pixels')

        overlap_px = int(tile_size * max(0.0, min(overlap_ratio, 0.45)))
        stride = max(1, tile_size - overlap_px)
        x_starts = get_tile_starts_with_stride(orig_w, tile_size, stride)
        y_starts = get_tile_starts_with_stride(orig_h, tile_size, stride)
        print(f'Tile overlap: {overlap_px}px ({overlap_ratio:.2f}) | Effective stride: {stride}px')

        mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
        total_tiles = len(x_starts) * len(y_starts)

        with tqdm(total=total_tiles, desc='Predicting tiles') as pbar:
            for y in y_starts:
                for x in x_starts:
                    window = Window(x, y, tile_size, tile_size)
                    w_real = min(tile_size, orig_w - x)
                    h_real = min(tile_size, orig_h - y)

                    img_bilinear = src.read([1, 2, 3], window=window,
                                            out_shape=(3, model_input_size, model_input_size),
                                            resampling=Resampling.bilinear,
                                            boundless=True, fill_value=0)

                    if np.max(img_bilinear) == 0:
                        pbar.update(1)
                        continue

                    img_nearest = src.read([1, 2, 3], window=window,
                                           out_shape=(3, model_input_size, model_input_size),
                                           resampling=Resampling.nearest,
                                           boundless=True, fill_value=0)
                    black_mask = (img_nearest[0] == 0) & (img_nearest[1] == 0) & (img_nearest[2] == 0)

                    max_val = 65535.0 if img_bilinear.dtype == np.uint16 else 255.0
                    img_norm = (img_bilinear.astype(np.float32) / max_val - IMAGE_MEAN) / IMAGE_STD
                    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits_oem = model(img_tensor)[0]
                        pred = torch.argmax(logits_oem, dim=1).cpu().numpy().squeeze().astype(np.uint8)

                    pred[black_mask] = 0
                    pred_upscaled = cv2.resize(pred, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

                    left_margin = overlap_px // 2 if x > 0 else 0
                    right_margin = overlap_px // 2 if (x + tile_size) < orig_w else 0
                    top_margin = overlap_px // 2 if y > 0 else 0
                    bottom_margin = overlap_px // 2 if (y + tile_size) < orig_h else 0

                    write_x0 = x + left_margin
                    write_x1 = x + w_real - right_margin
                    write_y0 = y + top_margin
                    write_y1 = y + h_real - bottom_margin

                    if write_x1 <= write_x0:
                        write_x0, write_x1 = x, x + w_real
                    if write_y1 <= write_y0:
                        write_y0, write_y1 = y, y + h_real

                    src_x0 = left_margin if write_x0 > x else 0
                    src_y0 = top_margin if write_y0 > y else 0
                    src_x1 = src_x0 + (write_x1 - write_x0)
                    src_y1 = src_y0 + (write_y1 - write_y0)

                    mask_full[write_y0:write_y1, write_x0:write_x1] = pred_upscaled[src_y0:src_y1, src_x0:src_x1]
                    pbar.update(1)

    output_path = f'{output_name}.png'
    save_palette_png(mask_full, output_path)
    print(f'\nSaved prediction to {output_path}')

    if preview:
        make_preview(tif_path, mask_full, output_name)
    return mask_full


def make_preview(tif_path, mask_full, output_name, downscale_factor=10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with rasterio.open(tif_path) as src:
        new_h = int(src.height / downscale_factor)
        new_w = int(src.width / downscale_factor)
        data = src.read(out_shape=(src.count, new_h, new_w), resampling=Resampling.bilinear)
    img = data[:3] if data.shape[0] >= 3 else data[0:1]
    img = np.transpose(img, (1, 2, 0)).astype(float)
    p2, p98 = np.percentile(img, (2, 98))
    img_normalized = np.clip((img - p2) / (p98 - p2), 0, 1)

    small_mask = cv2.resize(mask_full, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    rgb_mask = mask_to_rgb(small_mask)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=120)
    if img_normalized.shape[2] == 1:
        axes[0].imshow(img_normalized[:, :, 0], cmap='gray')
    else:
        axes[0].imshow(img_normalized)
    axes[0].set_title(f'Original Image (Downscaled {downscale_factor}x)')
    axes[0].axis('off')

    axes[1].imshow(rgb_mask)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    patches = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=np.array(c) / 255,
                          markersize=12, label=l) for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
    axes[1].legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    preview_path = f'{output_name}_preview.png'
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    print(f'Saved preview to {preview_path}')


def main():
    parser = argparse.ArgumentParser(description='Sliding-window prediction on a large geoTIFF with SAM2-UNet')
    parser.add_argument('--tif', type=str, required=True, help='Path to input geoTIFF')
    parser.add_argument('--weights', type=str, required=True, help='Path to fine-tuned SAM2-UNet weights (.pth)')
    parser.add_argument('--sam-checkpoint', type=str, required=True, help='Path to SAM2 backbone checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output filename without extension')
    parser.add_argument('--variant', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--resolution', type=float, default=None, help='Drone image GSD in meters/pixel (auto-detected if blank)')
    parser.add_argument('--target_gsd', type=float, default=TARGET_GSD, help='Target GSD used to pick crop size (default: 0.5 m/px)')
    parser.add_argument('--overlap', type=float, default=OVERLAP_RATIO, help='Tile overlap ratio in [0, 0.45]')
    parser.add_argument('--preview', action='store_true', help='Also save a side-by-side preview PNG')
    args = parser.parse_args()

    predict_mask_on_large_tif(
        args.tif, args.weights, args.sam_checkpoint, args.output,
        variant=args.variant,
        user_resolution=args.resolution,
        target_gsd=args.target_gsd,
        overlap_ratio=args.overlap,
        preview=args.preview,
    )


if __name__ == '__main__':
    main()