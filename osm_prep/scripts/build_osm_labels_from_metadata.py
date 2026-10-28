"""
Build OSM label masks for all images listed in two metadata CSV files.

This script merges the train and validation metadata into one table and
creates selected OSM masks per image (roads and buildings by default):
- roads:   highway class only
- buildings: building class only

Example:
    python build_osm_labels_from_metadata.py \
        --train-csv image_metadata_train_all.csv \
        --val-csv image_metadata_val_all.csv \
        --output-dir OSM_labels

Optional:
    --image-root C:\path\to\OpenEarthMap_wo_xBD
    --force-refresh
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time

import pandas as pd

from osm_patch_processor import OSMPatchProcessor, make_disk, thicken_mask

ROAD_CLASS = "highway"
BUILDING_CLASS = "building"
CLASS_KEYS = {'roads': ROAD_CLASS, 'buildings': BUILDING_CLASS, 'water': 'water', 'trees': 'trees'}
DEFAULT_OUTPUT_DIR = "OSM_labels"
DEFAULT_TIMEOUT_SECONDS = 120  # 2 minutes per patch

REQUIRED_COLUMNS = [
    "filename",
    "crs",
    "width_px",
    "height_px",
    "left",
    "bottom",
    "right",
    "top",
    "left_wgs84",
    "bottom_wgs84",
    "right_wgs84",
    "top_wgs84",
]


def load_metadata_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

    df = df.copy()
    df["filename"] = df["filename"].astype(str)
    df["filepath"] = df["filepath"].astype(str) if "filepath" in df.columns else ""
    return df


def merge_metadata(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train_df, val_df], ignore_index=True)

    # Prefer exact unique image rows. If the same image appears twice, keep the first row.
    combined = combined.drop_duplicates(subset=['filename'], keep="first").reset_index(drop=True)
    return combined


def ensure_output_structure(output_dir: str | Path, road_only: bool, classes=None) -> Path:
    output_dir = Path(output_dir)
    classes = classes if classes is not None else (['roads'] if road_only else ['roads', 'buildings'])
    for name in classes:
        (output_dir / name / 'binary_masks').mkdir(parents=True, exist_ok=True)
    return output_dir


def build_processor(metadata_df: pd.DataFrame, output_dir: str | Path, image_root: Optional[str | Path], verbose: bool) -> OSMPatchProcessor:
    return OSMPatchProcessor(
        patch_metadata=metadata_df,
        output_dir=output_dir,
        image_root=image_root,
        verbose=verbose,
    )


def iter_patch_ids(metadata_df: pd.DataFrame) -> Iterable[str]:
    return metadata_df["filename"].astype(str).tolist()


def get_patch_output_path(patch_filename: str, class_name: str, output_base: Path) -> Path:
    patch_stem = Path(str(patch_filename)).stem
    return output_base / patch_stem / f"{patch_stem}_{class_name}_mask.tif"


def is_patch_complete(patch_filename: str, class_names: list[str], output_base: Path) -> bool:
    for class_name in class_names:
        output_path = get_patch_output_path(patch_filename, class_name, output_base)
        if not output_path.exists():
            return False
    return True


class TimeoutError(Exception):
    pass


def download_with_timeout(processor, patch_id, semantic_keys, use_cache, cache_format, timeout_seconds):
    """Download OSM features with a timeout using threading."""
    result = {"gdf": None, "error": None}
    
    def _download():
        try:
            result["gdf"] = processor.download_patch_features(
                patch_id=patch_id,
                semantic_keys=semantic_keys,
                use_cache=use_cache,
                cache_format=cache_format,
            )
        except Exception as e:
            result["error"] = e
    
    thread = threading.Thread(target=_download, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        raise TimeoutError(f"OSM download timeout for {patch_id}")
    
    if result["error"]:
        raise result["error"]
    
    return result["gdf"]


def load_skip_list(skip_file: Path) -> set[str]:
    if not skip_file.exists():
        return set()
    with open(skip_file) as f:
        return set(line.strip() for line in f if line.strip())


def grid_record(processor, patch_id):
    profile = processor.get_patch_profile(patch_id)
    return {'crs': str(profile['crs']), 'width': profile['width'], 'height': profile['height'],
            'transform': list(profile['transform'])[:6], 'bounds_wgs84': processor.get_patch_bounds_wgs84(patch_id)}


def find_remaining_patches(metadata_df: pd.DataFrame, output_dir: Path, road_only: bool, classes=None, processor=None) -> list[str]:
    classes = classes if classes is not None else (['roads'] if road_only else ['roads', 'buildings'])
    remaining = []
    for patch_filename in metadata_df["filename"].astype(str).tolist():
        status_path = output_dir / 'status' / f'{Path(patch_filename).stem}.json'
        status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        if status.get('grid') is not None and processor is not None:
            if status['grid'] != grid_record(processor, patch_filename):
                raise ValueError(f'Grid changed for {patch_filename}; use a new --output-dir')
        complete = status.get('state') == 'success' and all(
            is_patch_complete(patch_filename, [CLASS_KEYS[name]], output_dir / name / 'binary_masks')
            and status.get('classes', {}).get(name, {}).get('state') == 'success'
            for name in classes
        )
        if not complete:
            remaining.append(patch_filename)
    return remaining


def export_class_from_patch_gdf(
    processor: OSMPatchProcessor,
    patch_id: str,
    patch_gdf,
    class_name: str,
    class_output_dir: Path,
    save_png: bool,
    road_buffer_pixels: int = 3,
):
    patch_stem = Path(str(patch_id)).stem
    output_dir = class_output_dir / patch_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = processor.export_binary_masks_for_patch(
        patch_id=patch_id,
        selected_keys=[class_name],
        patch_gdf=patch_gdf,
        output_dir=output_dir,
        save_png=save_png,
        all_touched=False,
        road_buffer_pixels=road_buffer_pixels,
    )
    return masks[class_name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build independent OSM class masks on the source tile grid.")
    parser.add_argument("--train-csv", required=True, help="Path to image_metadata_train_all.csv")
    parser.add_argument("--val-csv", required=True, help="Path to image_metadata_val_all.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Output root directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--image-root", default=None, help="Optional root folder containing the image TIFFs")
    parser.add_argument("--road", action="store_true", help="Generate only road masks and skip buildings")
    parser.add_argument('--classes', nargs='+', choices=list(CLASS_KEYS), help='Classes to export (default: roads buildings)')
    parser.add_argument('--road-buffer-pixels', type=int, default=3, help='Native-grid disk radius for roads; 0 keeps thin roads')
    parser.add_argument("--force-refresh", action="store_true", help="Redownload OSM even if cache exists")
    parser.add_argument("--no-png", action="store_true", help="Do not save PNG previews, only TIFFs")
    parser.add_argument("--keep-cache", action="store_true", help="Keep cache after running (speeds up reruns, uses ~1GB per 500 patches)")
    parser.add_argument("--no-cache", action="store_true", help="Skip caching entirely (saves disk space, no resumable cache)")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help=f"Timeout per patch in seconds (default: {DEFAULT_TIMEOUT_SECONDS})")
    parser.add_argument("--skip-file", default=None, help="Optional file with one patch filename per line to skip")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages")
    args = parser.parse_args()
    if args.road and args.classes is not None:
        parser.error('Use either --road or --classes')
    if args.road_buffer_pixels < 0:
        parser.error('--road-buffer-pixels must be nonnegative')
    classes = sorted(set(args.classes if args.classes is not None else (['roads'] if args.road else ['roads', 'buildings'])))

    train_df = load_metadata_csv(args.train_csv)
    val_df = load_metadata_csv(args.val_csv)
    metadata_df = merge_metadata(train_df, val_df)

    output_dir = ensure_output_structure(args.output_dir, road_only=args.road, classes=classes)
    config = {
        'filters': {name: OSMPatchProcessor.CLASS_TAGS[CLASS_KEYS[name]] for name in classes},
        'road_buffer_pixels': args.road_buffer_pixels, 'all_touched': False,
        'encoding': 'uint8_0_1', 'grid': 'source_tiff_or_exact_affine', 'version': 1,
    }
    config_path = output_dir / 'preparation_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Preparation settings changed; use a new --output-dir')
    else:
        if any(path.is_file() and path.suffix.lower() in ('.tif', '.tiff') for path in output_dir.rglob('*')):
            raise FileExistsError('Existing masks have no configuration record; use a new --output-dir')
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    status_dir = output_dir / 'status'
    status_dir.mkdir(exist_ok=True)
    cache_root = output_dir / "_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Train rows: {len(train_df):,}")
        print(f"Val rows:   {len(val_df):,}")
        print(f"Merged rows: {len(metadata_df):,}")
        print(f"Output dir:  {output_dir.resolve()}")
        print(f"Classes:     {', '.join(classes)}")
        cache_mode = 'disabled' if args.no_cache else ('kept' if args.keep_cache else 'deleted after')
        print(f"Cache:       {cache_mode}")

    save_png = not args.no_png
    total = len(metadata_df)
    success = 0
    failed = 0

    processor = build_processor(
        metadata_df=metadata_df,
        output_dir=cache_root,
        image_root=args.image_root,
        verbose=args.verbose,
    )

    selected_keys = [CLASS_KEYS[name] for name in classes]

    remaining_patches = find_remaining_patches(metadata_df, output_dir, args.road, classes=classes, processor=processor)
    skip_list = load_skip_list(Path(args.skip_file)) if args.skip_file else set()
    remaining_patches = [p for p in remaining_patches if p not in skip_list]
    
    if args.verbose:
        skipped = total - len(remaining_patches)
        if skipped > 0:
            print(f"Found {skipped:,} already-processed patch(es), resuming from {len(remaining_patches):,} remaining")
        if skip_list:
            print(f"Skipping {len(skip_list):,} patch(es) from skip-file")
        print()

    try:
        for idx, patch_id in enumerate(remaining_patches, start=1):
            status_path = status_dir / f'{Path(patch_id).stem}.json'
            status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'classes': {}}
            try:
                status['grid'] = grid_record(processor, patch_id)
                pending = []
                for name in classes:
                    path = get_patch_output_path(patch_id, CLASS_KEYS[name], output_dir / name / 'binary_masks')
                    if path.exists():
                        if status['classes'].get(name, {}).get('state') != 'success':
                            raise FileExistsError(f'Existing mask has no successful status: {path}; use a new output directory')
                    else:
                        pending.append(name)
                if args.verbose:
                    print(f"[{idx}/{len(remaining_patches)}] {patch_id} -> download")

                status['query'] = 'pending'
                patch_gdf = download_with_timeout(
                    processor=processor,
                    patch_id=patch_id,
                    semantic_keys=selected_keys,
                    use_cache=not (args.force_refresh or args.no_cache),
                    cache_format="gpkg",
                    timeout_seconds=args.timeout_seconds,
                )

                status['query'] = 'success_empty' if patch_gdf.empty else 'success_nonempty'
                for name in pending:
                    if args.verbose:
                        print(f"[{idx}/{len(remaining_patches)}] {patch_id} -> {name} mask")
                    mask = export_class_from_patch_gdf(
                        processor=processor,
                        patch_id=patch_id,
                        patch_gdf=patch_gdf,
                        class_name=CLASS_KEYS[name],
                        class_output_dir=output_dir / name / 'binary_masks',
                        save_png=save_png,
                        road_buffer_pixels=args.road_buffer_pixels,
                    )
                    status['classes'][name] = {'state': 'success', 'empty': not bool(mask.any())}
                    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')

                status['state'] = 'success'
                status.pop('error', None)
                status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
                success += 1
                if args.verbose:
                    print(f"[{idx}/{len(remaining_patches)}] done: {patch_id}")
            except TimeoutError as exc:
                status.update(state='failure', query='timeout', error=str(exc))
                status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
                failed += 1
                print(f"[TIMEOUT] {patch_id}: {exc}")
                print(f"  Tip: add '{patch_id}' to skip-file to skip in future runs")
            except Exception as exc:
                status.update(state='failure', error=str(exc))
                if status.get('query') in (None, 'pending'):
                    status['query'] = 'failure'
                status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
                failed += 1
                exc_str = str(exc)
                if "cannot convert float NaN" in exc_str or "invalid value" in exc_str:
                    print(f"[GEOMETRY ERROR] {patch_id}: Invalid coordinate/geometry")
                elif "Expecting value" in exc_str or "JSON" in exc_str:
                    print(f"[API ERROR] {patch_id}: Overpass API returned invalid response")
                else:
                    print(f"[ERROR] {patch_id}: {exc}")
    finally:
        if not args.keep_cache:
            shutil.rmtree(cache_root, ignore_errors=True)

    print()
    print("Finished.")
    print(f"Successful patches: {success:,}")
    print(f"Failed patches:     {failed:,}")
    print(f"Labels written to:   {output_dir.resolve()}")
    cache_status = "kept" if args.keep_cache else ("not created" if args.no_cache else "deleted")
    print(f"Cache:               {cache_status}")
    if args.keep_cache:
        print(f"Cache location:      {cache_root.resolve()}")
    print("Per patch output layout:")
    for name in classes:
        print(f'  {name}/binary_masks/<patch>/<patch>_{CLASS_KEYS[name]}_mask.tif')


if __name__ == "__main__":
    main()
