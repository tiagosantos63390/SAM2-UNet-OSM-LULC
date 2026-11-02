"""Predict overlapping RGB tiles at a fixed physical scale; export PNG and GeoTIFF."""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp, Resampling
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.checkpoints import load_model  # noqa: E402

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


def get_tile_starts_with_stride(dimension_length, tile_size, stride):
    stride = max(1, min(stride, tile_size))
    starts = [0]
    while starts[-1] + tile_size < dimension_length:
        starts.append(starts[-1] + stride)
    return starts


def target_grid(src, target_gsd):
    if not math.isfinite(target_gsd) or target_gsd <= 0:
        raise ValueError('Target GSD must be finite and positive')
    if src.crs is None or not src.crs.is_projected:
        raise ValueError('A projected CRS with known linear units is required; reproject geographic imagery first')
    try:
        _, metres_per_unit = src.crs.linear_units_factor
    except (ValueError, rasterio.errors.CRSError) as exc:
        raise ValueError('Cannot convert CRS linear units to metres') from exc
    t = src.transform
    if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
        raise ValueError('A north-up, unrotated raster with positive spatial resolution is required')
    if not math.isfinite(metres_per_unit) or metres_per_unit <= 0:
        raise ValueError('CRS linear unit conversion must be finite and positive')
    resolution = target_gsd / metres_per_unit
    bounds = src.bounds
    if not all(math.isfinite(v) for v in bounds):
        raise ValueError('Raster bounds must be finite')
    return (from_origin(bounds.left, bounds.top, resolution, resolution),
            math.ceil((bounds.right - bounds.left) / resolution),
            math.ceil((bounds.top - bounds.bottom) / resolution))


def mask_to_rgb(mask):
    """Map a 2D class mask to RGB using a lookup table (chunked to save memory)."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:9] = CLASS_COLORS
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

    if not 0 <= overlap_ratio <= .45:
        raise ValueError('Overlap must be in [0, 0.45]')
    if model_input_size != MODEL_INPUT_SIZE:
        raise ValueError('This model expects 512-pixel inputs')
    with rasterio.open(tif_path) as src:
        transform, width, height = target_grid(src, target_gsd)
        if src.count < 3 or any(dtype not in ('uint8', 'uint16') for dtype in src.dtypes[:3]):
            raise ValueError('Expected at least three RGB bands encoded as uint8 or uint16')
        if user_resolution is not None:
            native_x = src.res[0] * src.crs.linear_units_factor[1]
            native_y = src.res[1] * src.crs.linear_units_factor[1]
            if not all(math.isclose(user_resolution, value, rel_tol=1e-5) for value in (native_x, native_y)):
                raise ValueError('--resolution conflicts with georeferencing; correct the source metadata first')
        model = load_model(weights_path, sam_checkpoint, variant, device)
        overlap_px = int(model_input_size * overlap_ratio)
        stride = model_input_size - overlap_px
        x_starts = get_tile_starts_with_stride(width, model_input_size, stride)
        y_starts = get_tile_starts_with_stride(height, model_input_size, stride)
        mask_full = np.full((height, width), 255, dtype=np.uint8)
        # The VRT resamples onto an exact physical grid before any tile is read.
        with WarpedVRT(src, crs=src.crs, transform=transform, width=width, height=height,
                       resampling=Resampling.bilinear,
                       add_alpha=ColorInterp.alpha not in src.colorinterp) as grid:
            for y in tqdm(y_starts, desc='Predicting tile rows'):
                for x in x_starts:
                    w_real = min(model_input_size, width - x)
                    h_real = min(model_input_size, height - y)
                    window = Window(x, y, w_real, h_real)
                    image = np.zeros((3, model_input_size, model_input_size), dtype=np.float32)
                    valid = np.zeros((model_input_size, model_input_size), dtype=bool)
                    image[:, :h_real, :w_real] = grid.read([1, 2, 3], window=window)
                    valid[:h_real, :w_real] = grid.dataset_mask(window=window) > 0
                    if not valid.any():
                        continue
                    max_val = 65535.0 if src.dtypes[0] == 'uint16' else 255.0
                    normalized = (image / max_val - IMAGE_MEAN) / IMAGE_STD
                    normalized[:, ~valid] = 0
                    tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        pred = model.forward_semantic(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
                    pred[~valid] = 255
                    left = overlap_px // 2 if x > 0 else 0
                    top = overlap_px // 2 if y > 0 else 0
                    right = (overlap_px + 1) // 2 if x + model_input_size < width else 0
                    bottom = (overlap_px + 1) // 2 if y + model_input_size < height else 0
                    x_end, y_end = w_real - right, h_real - bottom
                    mask_full[y + top:y + y_end, x + left:x + x_end] = pred[top:y_end, left:x_end]
        with rasterio.open(f'{output_name}.tif', 'w', driver='GTiff', height=height,
                           width=width, count=1, dtype='uint8', crs=src.crs,
                           transform=transform, nodata=255, compress='deflate') as dst:
            dst.write(mask_full, 1)
            dst.update_tags(target_gsd_metres=target_gsd, semantic_classes=9,
                            checkpoint=str(weights_path), inference_path='semantic_only')
    save_palette_png(mask_full, f'{output_name}.png')
    if preview:
        make_preview(tif_path, mask_full, output_name)
    return mask_full


def make_preview(tif_path, mask_full, output_name, downscale_factor=10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with rasterio.open(tif_path) as src:
        new_h = max(1, int(src.height / downscale_factor))
        new_w = max(1, int(src.width / downscale_factor))
        data = src.read(out_shape=(src.count, new_h, new_w), resampling=Resampling.bilinear)
    img = data[:3] if data.shape[0] >= 3 else data[0:1]
    img = np.transpose(img, (1, 2, 0)).astype(float)
    p2, p98 = np.percentile(img, (2, 98))
    img_normalized = np.clip((img - p2) / max(p98 - p2, 1e-6), 0, 1)

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
    plt.close(fig)
    print(f'Saved preview to {preview_path}')


def main():
    parser = argparse.ArgumentParser(description='Sliding-window prediction on a large geoTIFF with SAM2-UNet')
    parser.add_argument('--tif', type=str, required=True, help='Path to input geoTIFF')
    parser.add_argument('--weights', type=str, required=True, help='Path to fine-tuned SAM2-UNet weights (.pth)')
    parser.add_argument('--sam-checkpoint', type=str, required=True, help='Path to SAM2 backbone checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output filename without extension')
    parser.add_argument('--variant', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--resolution', type=float, default=None, help='Check native GSD in metres/pixel against raster metadata')
    parser.add_argument('--target-gsd', '--target_gsd', type=float, default=TARGET_GSD, help='Metres per model-input pixel (default: 0.5)')
    parser.add_argument('--overlap', type=float, default=OVERLAP_RATIO, help='Tile overlap ratio in [0, 0.45]')
    parser.add_argument('--device', default=None)
    parser.add_argument('--preview', action='store_true', help='Also save a stacked RGB/prediction preview PNG')
    args = parser.parse_args()

    predict_mask_on_large_tif(
        args.tif, args.weights, args.sam_checkpoint, args.output,
        variant=args.variant,
        user_resolution=args.resolution,
        target_gsd=args.target_gsd,
        overlap_ratio=args.overlap,
        preview=args.preview,
        device=args.device,
    )


if __name__ == '__main__':
    main()
