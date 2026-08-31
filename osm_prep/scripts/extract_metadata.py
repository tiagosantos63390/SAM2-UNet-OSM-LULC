"""
Extract image metadata from GeoTIFF files and generate CSV rows.

This script scans a folder for GeoTIFF images and extracts:
- filename
- filepath
- CRS
- image dimensions (width, height, bands)
- dtype
- pixel size
- projected bounds (left, bottom, right, top)
- center coordinates
- WGS84 bounds

Usage:
    # Extract all validation images using val.txt
    python extract_metadata.py --input-folder "path/to/OpenEarthMap" --output-csv metadata_val.csv --filter "path/to/val.txt"
    
    # Extract images matching filename pattern
    python extract_metadata.py --input-folder "path/to/OpenEarthMap" --output-csv metadata_new.csv --filter "val_"
    
    # Extract all images (no filter)
    python extract_metadata.py --input-folder "path/to/OpenEarthMap" --output-csv metadata_all.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import rasterio
from rasterio.crs import CRS
import warnings

warnings.filterwarnings('ignore')


def extract_image_metadata(image_path):
    """Extract metadata from a single GeoTIFF image."""
    try:
        with rasterio.open(image_path) as src:
            # Basic raster info
            width = src.width
            height = src.height
            count = src.count  # number of bands
            dtype = src.dtypes[0]
            
            # Spatial info
            crs = src.crs
            bounds = src.bounds  # (left, bottom, right, top)
            transform = src.transform
            
            # Pixel size (from transform)
            pixel_size_x = abs(transform.a)
            pixel_size_y = abs(transform.e)
            
            # Center in projected coords
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2
            
            # Convert to WGS84 if not already
            if crs and crs != CRS.from_epsg(4326):
                # Reproject bounds to WGS84
                from rasterio.warp import transform_bounds
                left_wgs84, bottom_wgs84, right_wgs84, top_wgs84 = transform_bounds(
                    crs, CRS.from_epsg(4326),
                    bounds.left, bounds.bottom, bounds.right, bounds.top
                )
                
                # Reproject center point
                from rasterio.warp import transform
                xs, ys = transform(crs, CRS.from_epsg(4326), [center_x], [center_y])
                center_lon, center_lat = xs[0], ys[0]
            else:
                left_wgs84, bottom_wgs84, right_wgs84, top_wgs84 = bounds.left, bounds.bottom, bounds.right, bounds.top
                center_lon, center_lat = center_x, center_y
            
            crs_str = str(crs) if crs else "Unknown"
            
            return {
                "filename": image_path.name,
                "filepath": str(image_path),
                "crs": crs_str,
                "width_px": width,
                "height_px": height,
                "bands": count,
                "dtype": dtype,
                "pixel_size_x": pixel_size_x,
                "pixel_size_y": pixel_size_y,
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
                "center_x": center_x,
                "center_y": center_y,
                "center_lon": center_lon,
                "center_lat": center_lat,
                "left_wgs84": left_wgs84,
                "bottom_wgs84": bottom_wgs84,
                "right_wgs84": right_wgs84,
                "top_wgs84": top_wgs84,
            }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def load_image_list_from_file(txt_file_path):
    """Load image list from a .txt file (e.g., val.txt)."""
    try:
        with open(txt_file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(lines)} image names from {txt_file_path}")
        return set(lines)
    except Exception as e:
        print(f"Error reading {txt_file_path}: {e}")
        return None


def extract_metadata_from_folder(folder_path, pattern="*.tif", recursive=True, txt_filter_path=None):
    """Scan folder and extract metadata from all matching images."""
    folder = Path(folder_path)
    
    # Load validation list if provided
    validation_set = None
    if txt_filter_path:
        txt_path = Path(txt_filter_path)
        if txt_path.exists():
            validation_set = load_image_list_from_file(txt_path)
    
    if recursive:
        image_paths = list(folder.rglob(pattern))
    else:
        image_paths = list(folder.glob(pattern))
    
    print(f"Found {len(image_paths)} images matching pattern '{pattern}'")
    
    # Filter by validation set if provided
    if validation_set:
        image_paths = [p for p in image_paths if p.name in validation_set or p.stem in validation_set]
        print(f"After filtering by validation list: {len(image_paths)} images")
    
    metadata_list = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
        meta = extract_image_metadata(image_path)
        if meta:
            metadata_list.append(meta)
    
    return pd.DataFrame(metadata_list)


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata from GeoTIFF images for OSM processing."
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to folder containing images (e.g., OpenEarthMap dataset)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="extracted_metadata.csv",
        help="Output CSV filename (default: extracted_metadata.csv)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.tif",
        help="File pattern to match (default: *.tif)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Optional filter: either a val.txt file path (e.g., 'path/to/val.txt') or a filename substring"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively search subfolders (default: True)"
    )
    
    args = parser.parse_args()
    
    print(f"Scanning folder: {args.input_folder}")
    
    # Determine if filter is a file path or a substring
    txt_filter_path = None
    substring_filter = None
    
    if args.filter:
        filter_path = Path(args.filter)
        if filter_path.exists() and filter_path.is_file():
            # It's a file path (e.g., val.txt)
            txt_filter_path = args.filter
            print(f"Using validation list from: {args.filter}")
        else:
            # Check if "val.txt" or similar exists in the input folder
            input_folder = Path(args.input_folder)
            possible_val_file = input_folder / f"{args.filter}.txt"
            if possible_val_file.exists():
                txt_filter_path = str(possible_val_file)
                print(f"Found validation list at: {txt_filter_path}")
            else:
                # It's a substring filter
                substring_filter = args.filter
                print(f"Filtering for filenames containing: {args.filter}")
    
    # Extract metadata
    df = extract_metadata_from_folder(
        args.input_folder,
        pattern=args.pattern,
        recursive=args.recursive,
        txt_filter_path=txt_filter_path
    )
    
    # Apply substring filter if specified (and no txt file was used)
    if substring_filter and not txt_filter_path:
        df = df[df["filename"].str.contains(substring_filter, case=False, na=False)]
        print(f"After filter: {len(df)} images")
    
    # Save to CSV
    output_path = Path(args.output_csv)
    df.to_csv(output_path, index=False)
    print(f"\nMetadata saved to: {output_path}")
    print(f"Total images: {len(df)}")
    
    # Show sample
    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
