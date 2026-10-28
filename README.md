# SAM2-UNet with OpenStreetMap auxiliary supervision

This repository contains the training and evaluation code for the EPIA 2026
paper on high-resolution land use / land cover semantic segmentation:
*"Adapting SAM2-UNet for High-Resolution LULC Semantic Segmentation with
OpenStreetMap Auxiliary Supervision"*.

The model is a dual-decoder adaptation of SAM2-UNet (SAM2 Hiera-tiny encoder +
receptive-field blocks + decoder) trained on [OpenEarthMap](#data). A second
decoder head is supervised during training with OpenStreetMap-derived road and
building masks used as distance-transform soft targets (MSE loss, `lambda_road
= 0.5`, `lambda_building = 0.25`, max distance `d = 6`). The auxiliary head
outputs 2 channels (channel 0 = roads, channel 1 = buildings). At inference
time the auxiliary branch is dropped, so the model runs on RGB imagery only.

## Repository contents

```
├── src/
│   ├── __init__.py
│   ├── model.py        # DualDecoder_SAM2UNet: frozen SAM2 encoder, adapters, RFB blocks, dual decoders
│   ├── dataset.py      # OpenEarthMap tiles + OSM road/building mask loading and augmentation
│   └── metrics.py      # mIoU, precision/recall, F1, per-class IoU, Boundary IoU
├── scripts/
│   ├── train.py                  # Baseline + auxiliary training runs over multiple seeds
│   ├── evaluate.py               # Evaluate a trained checkpoint on the test split
│   ├── predict.py                # Single-image inference -> indexed PNG
│   ├── predict_large_tif.py      # Sliding-window inference on large geoTIFFs (UAV orthomosaics)
│   ├── aggregate_metrics.py      # Mean/std over seed metrics JSONs (produces the results table)
│   └── plot_results.py           # Renders the test comparison chart from the metric JSONs
├── data/splits/                  # Exact train/val/test split files (1879 / 404 / 404 tiles)
├── osm_prep/                     # OpenStreetMap road/building mask rasterisation pipeline
├── notebooks/                    # Original Kaggle notebooks used for the reported experiments
├── figures/                      # Figures referenced below
└── results/
    ├── history/                  # Per-seed test metrics (seeds 42, 43, 44)
    └── test_comparison.png
```

The modules in `src/` and `scripts/` are taken from the notebooks (only the
Kaggle-specific paths were adapted for local use), so the released code matches
what was actually run. The figure below shows how rasterised OSM road masks
align with the OpenEarthMap annotations.

![OSM and OpenEarthMap annotation alignment](figures/alignment_oem_osm.png)

## Notebooks

- `notebooks/train_paper_models_seeds42-44.ipynb` trains all five
  configurations (baseline, roads only at lambda 0.5, roads+buildings at
  lambda_building 0.25/0.50/0.75) over seeds 42, 43 and 44. It writes one
  `{run}_seed{seed}_metrics.json` and one `{run}_seed{seed}_final.pth` per run;
  Kaggle's session limit means each seed was run in its own session.
- `notebooks/dt_ablation_lambda0p5_d3-6-10.ipynb` contains the distance-transform
  ablation on the validation split (d = 3, 6, 10 at lambda 0.5), plus the
  no-OSM baseline.

## Setup

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git
```

You also need the SAM2 tiny encoder checkpoint (`sam2_hiera_tiny.pt`) from the
[SAM2 releases](https://github.com/facebookresearch/sam2), e.g.:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

## Data

The model is trained and evaluated on **OpenEarthMap** (excluding the xBD
subset), organised as `<city>/images/*.tif` and `<city>/labels/*.tif`. You can
obtain it from:

- our Kaggle copy: https://www.kaggle.com/datasets/tiagosantos71921/openearthmap-thesis
- the official release: https://open-earth-map.org/ (also on Zenodo, https://zenodo.org/records/7223446)

The exact train / validation / test tile lists used for the paper are in
`data/splits/` (`train.txt` 1879 tiles, `val.txt` 404, `test.txt` 404), one
tile name per line. They match the official OpenEarthMap split.

### OSM auxiliary labels

Rasterised binary road (``_highway_mask.tif``) and building
(``_building_mask.tif``) masks aligned to the tiles are needed only for the
auxiliary training runs. Two options:

1. **Download pre-generated masks** from our Kaggle dataset
   (includes the per-tile metadata CSVs):
   https://www.kaggle.com/datasets/tiagosantos71921/openstreetmap
2. **Generate them yourself** with the rasterisation pipeline in
   [`osm_prep/`](osm_prep/README.md) (OSM tags `highway=*` / `building=*`,
   roads rasterised and thickened to ~6 px, per-tile UTM grids). See its
   README for the exact commands.

## Training

```bash
python scripts/train.py \
    --data-root /path/to/OpenEarthMap_wo_xBD \
    --train-txt data/splits/train.txt --val-txt data/splits/val.txt --test-txt data/splits/test.txt \
    --osm-roads-dir /path/to/osm/roads/binary_masks \
    --osm-buildings-dir /path/to/osm/buildings/binary_masks \
    --sam-checkpoints-dir /path/to/sam-checkpoints \
    --seeds 42 43 44
```

Training uses AdamW (lr 1e-3, weight decay 5e-4), a cosine schedule, batch
size 4 on 512x512 tiles, mixed precision, and early stopping on validation
mIoU. Each run's best checkpoint is saved as `{run}_seed{seed}_final.pth`
under `outputs/`, and its test metrics under `history/`.

Use `--experiments NAME [NAME ...]` to select runs; the default five are unchanged.

| Name | Auxiliary losses |
| --- | --- |
| `bldg_only_L0p25_bce_dice` | Building: 0.25 × BCE+Dice |
| `bldg_only_L0p5_bce_dice` | Building: 0.50 × BCE+Dice |
| `four_class_L0p5_d6_dt_mse` | Road/building/water/tree: 0.50 × DT-MSE each, d=6 |
| `four_class_L0p5_d6_mixed` | Same weights; building uses BCE+Dice, others DT-MSE |

Four-class runs require `--osm-water-dir` and `--osm-trees-dir`, each pointing
to its mask directory. Auxiliary channels are road=0, building=1, water=2,
tree=3; other runs retain two channels. BCE+Dice weights both terms equally,
uses Dice smoothing of 1e-6, and excludes invalid supervision. These
configurations are not verified reproductions of historical runs.
Use a new `--work-dir` if run outputs already exist.

## Evaluation

```bash
python scripts/evaluate.py \
    --data-root /path/to/OpenEarthMap_wo_xBD \
    --split-txt data/splits/test.txt \
    --sam-checkpoint /path/to/sam2_hiera_tiny.pt \
    --weights /path/to/roads_bldg_L0p25_seed42_final.pth
```

Prints mIoU, F1, precision/recall, Boundary IoU, road IoU and the per-class
IoU breakdown, using the same evaluation function as training.

## Prediction

Single image (any format, resized to 512 and ImageNet-normalised):

```bash
python scripts/predict.py \
    --image /path/to/image.tif \
    --sam-checkpoint /path/to/sam2_hiera_tiny.pt \
    --weights /path/to/roads_bldg_L0p25_seed42_final.pth \
    --output prediction.png
```

Large geoTIFF (e.g. a UAV / drone orthomosaic). The tile size is derived from
the image GSD so each tile covers the same ~0.5 m/px physical footprint as the
OpenEarthMap training data; overlapping tiles are stitched and exported as an
indexed PNG using the 9-class OpenEarthMap palette:

```bash
python scripts/predict_large_tif.py \
    --tif /path/to/uav_orthomosaic.tif \
    --sam-checkpoint /path/to/sam2_hiera_tiny.pt \
    --weights /path/to/roads_bldg_L0p25_seed42_final.pth \
    --output uav_prediction
```

(Optionally pass `--resolution <GSD>` when the tif metadata resolution is
missing, and `--preview` to also save a side-by-side image + prediction PNG.)

## Results

OpenEarthMap test split, mean +/- std over seeds 42, 43, 44:

| Model | mIoU | F1 | Boundary mIoU |
|---|---|---|---|
| Baseline SAM2-UNet | 0.6545 +/- 0.0101 | 0.7811 +/- 0.0075 | 0.5643 +/- 0.0132 |
| OSM Auxiliary DT (Roads Only) | **0.6594 +/- 0.0074** | **0.7846 +/- 0.0049** | **0.5698 +/- 0.0095** |
| OSM Auxiliary DT (Roads+Buildings) | 0.6536 +/- 0.0076 | 0.7797 +/- 0.0070 | 0.5675 +/- 0.0065 |

To reproduce these numbers from the metric files:

```bash
cd results && python ../scripts/aggregate_metrics.py
```

To regenerate the comparison chart:

```bash
python scripts/plot_results.py
```

![Test set comparison](results/test_comparison.png)

Distance-transform ablation (validation split): the d=6 setting gives the best
validation behaviour (mIoU 0.6777, Boundary mIoU 0.5727, Road IoU 0.5777)
compared with d=3, d=10, and the no-OSM baseline. See the ablation notebook
for full outputs.

Auxiliary supervision mainly helps boundary quality on thin structures such as
roads. The qualitative figure below contrasts baseline predictions with the
roads-only auxiliary model.

![Predictions](figures/sam2unet_predictions_confidence.png)

## Pre-trained weights

Trained checkpoints are available for download:
https://github.com/tiagosantos63390/SAM2-UNet-OSM-LULC/releases

This is the `roads_bldg_L0p25` (roads+buildings, lambda_building = 0.25,
d = 6) configuration, matching the README results table. Weights are a plain
`state_dict` for `DualDecoder_SAM2UNet` (trained with a 2-channel OSM auxiliary
head); load them with `evaluate.py` / `predict.py` / `predict_large_tif.py`
after building the model with the corresponding SAM2 backbone checkpoint.

## License

The code in this repository is released under the
[Apache License 2.0](LICENSE).

## Acknowledgments

This work builds on the following open-source projects and datasets:

- **SAM 2**: N. Ravi, V. Gabeur, Y.-T. Hu, et al., "SAM 2: Segment Anything in
  Images and Videos", *arXiv:2408.00714*, 2024.
  Code: https://github.com/facebookresearch/sam2 (Apache-2.0)
- **SAM2-UNet**: X. Xiong, Z. Wu, S. Tan, et al., "SAM2-UNet: Segment Anything
  2 Makes Strong Encoder for Natural and Medical Image Segmentation",
  *arXiv:2408.08870*, 2024.
  Code: https://github.com/WZH0120/SAM2-UNet (Apache-2.0)
- **OpenEarthMap**: J. Xia, N. Yokoya, B. Adriano, et al., "OpenEarthMap: A
  Benchmark Dataset for Global High-Resolution Land Cover Mapping", *WACV*
  2023. Dataset: https://open-earth-map.org/ (CC-BY-4.0)
- **OpenStreetMap**: OpenStreetMap contributors, spatially referenced from
  OSMnx queries of the live OSM database. See M. Haklay and P. Weber,
  "OpenStreetMap: User-Generated Street Maps", *IEEE Pervasive Computing* 7(4),
  2008. https://www.openstreetmap.org (ODbL)