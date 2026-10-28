# OSM rasterisation preprocessing

This folder contains the exact scripts used to build the OpenStreetMap auxiliary
labels (road centerlines and building footprints) aligned onto the OpenEarthMap
tiles used to train the SAM2-UNet + OSM model. The training code lives in the
main repository; this folder only covers label generation.

## What it does

- `scripts/extract_metadata.py` scans an OpenEarthMap folder and writes one
  metadata CSV row per tile (projected bounds, WGS84 bounds, CRS, pixel size).
- `scripts/osm_patch_processor.py` is the core class: it downloads OSM features
  for a tile bounding box with `osmnx`, reprojects them to the tile CRS and
  rasterises them on the tile's native pixel grid.
- `scripts/build_osm_labels_from_metadata.py` merges the train/val metadata
  tables and writes, for every tile, a binary road mask (`*_highway_mask.tif`)
  and a binary building mask (`*_building_mask.tif`).

## Parameters (as used for the paper)

- OSM tags are presence-only: roads use `highway=*`, buildings use `building=*`.
- Roads are rasterised at their natural 1 px width and then thickened with a
  discrete disk structuring element of radius 3 px (~6 px net line width).
- Every tile is rasterised independently on its native projected grid (per-city
  UTM zone, e.g. EPSG:25832 for Aachen); because neighbouring tiles are
  contiguous UTM squares, roads stay continuous across tile borders.
- The model input size (512x512) is applied later at training time, not here.

## Setup

```bash
pip install -r scripts/requirements.txt
```

## Usage

```bash
# 1. Build the per-tile metadata tables from the OpenEarthMap folder
python scripts/extract_metadata.py \
    --input-folder /path/to/OpenEarthMap_wo_xBD \
    --output-csv metadata/image_metadata_train_all.csv \
    --filter data/splits/train.txt
python scripts/extract_metadata.py \
    --input-folder /path/to/OpenEarthMap_wo_xBD \
    --output-csv metadata/image_metadata_val_all.csv \
    --filter data/splits/val.txt

# 2. Download OSM and rasterise road + building masks
python scripts/build_osm_labels_from_metadata.py \
    --train-csv metadata/image_metadata_train_all.csv \
    --val-csv metadata/image_metadata_val_all.csv \
    --output-dir OSM_labels
```

Output layout:

```
OSM_labels/
  roads/binary_masks/<tile>/<tile>_highway_mask.tif
  buildings/binary_masks/<tile>/<tile>_building_mask.tif
```

Pass these folders to the training script with `--osm-roads-dir` and
`--osm-buildings-dir`.

### Optional water and tree masks

Use `--classes roads buildings water trees` to export all four classes, or
select a subset. The default remains roads + buildings. `--road` selects roads
alone and cannot be combined with `--classes`.

Water uses `natural` in `{water, waterway}` OR `landuse` in `{reservoir, basin}`.
Trees use `natural` in `{wood, scrub, tree_row}` OR `landuse=forest`. These are
the v22 predicates; `natural=waterway` is literal. Query values are combined,
and masks may overlap. Roads require `highway`; buildings require `building`.

`--road-buffer-pixels` defaults to a disk radius of 3 native-grid pixels, applied
once after road rasterization. Use `0` for thin roads. Other classes are not
dilated; `all_touched=False` applies to all classes. Masks are uint8 0/1 TIFFs:
`water/binary_masks/<tile>/<tile>_water_mask.tif` and
`trees/binary_masks/<tile>/<tile>_trees_mask.tif`. PNG previews go in `previews/`.

Grids preserve source dimensions, CRS and affine transform. Provide source TIFFs
with `--image-root` or metadata containing `transform_a` through `transform_f`.
Bounds alone are insufficient; missing or inconsistent grid information is rejected.

`preparation_config.json` and per-tile `status/*.json` record settings, completed
classes, empty queries and failures. Existing masks are never overwritten. Use
a fresh output directory for unrecorded masks or changed class, filter, grid or
dilation settings. `--force-refresh` refreshes pending queries, not completed masks.
Historical datasets are not regenerated or relabelled.

## Pre-generated masks

The ready-to-use rasterised masks (and the per-tile metadata CSVs above) are
also published as a Kaggle dataset:
https://www.kaggle.com/datasets/tiagosantos71921/openstreetmap

## Notes

- The metadata CSVs describe the tile set the OSM labels were generated for
  (2303 train + 384 val rows); they are a superset of the split lists in
  `data/splits/` (1879 / 404 / 404), so training will only use the tiles
  present in the split files.
- Downloads use OpenStreetMap/Overpass through `osmnx` and are cached per tile;
  runs are resumable and failed patches are recorded in skip files.