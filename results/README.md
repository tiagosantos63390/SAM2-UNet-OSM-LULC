# Experiment results

`history/` contains the paper's per-seed summaries. `historical/` contains the
additional saved metrics and distance-sweep histories. `index.json` records
paths, seeds, notebook versions, file hashes and available checkpoint details.
Original result files are kept unchanged.

| Notebook family (`sam2unet-train-kaggle-`) | Versions | Runs |
| --- | --- | --- |
| `paper-models` | 6 | Baseline, roads-only and roads/buildings DT-MSE |
| `paper-models-bce-dice` | 2/3/4 | Building-weight validation sweep with BCE+Dice |
| `paper-models-bce-dice` | 8 | Buildings-only BCE+Dice, weights 0.25 and 0.50 |
| `distance-transform3` | 5 | Baseline and d=3/6/10, seed 42 |
| `double-classifier-r-b-t-w` | 3/4/5 | Four-class mixed losses, seeds 42/43/44 |

Run aggregation from the repository root:

```bash
python scripts/aggregate_metrics.py --stage test --ddof 0 --output test_summary.json
python scripts/aggregate_metrics.py --stage best-val --ddof 1 \
    --family sam2unet-train-kaggle-paper-models-bce-dice --output building_sweep.json
python scripts/aggregate_metrics.py --stage best-val \
    --family sam2unet-train-kaggle-distance-transform3 --output distance_sweep.json
```

`test` and `val` read the saved final evaluations. `best-val` uses the
maximum-mIoU history row or a saved best-validation summary. The building-weight
summaries retain the four decimal places printed in the original logs. Where
only best mIoU was saved, other metrics are omitted. Population SD (`ddof=0`)
is the default; `--ddof 1` selects sample SD, undefined for a single run.
All calculations use the available per-run precision without adjusting values
to match rounded tables.

These results retain their original training settings. The distance sweep
used an earlier 1880/403/404 split and logged no matched OSM masks. Four-class
v3/v4/v5 used building BCE+Dice and an interior distance transform; the current
DT target uses distance to foreground. Historical best-validation summaries
and final checkpoint evaluations can refer to different epochs. These results
therefore remain separate from outputs of the current training scripts.

For new runs, use `--history-dir /path/to/run/history`. Their saved `val`
metrics describe the selected checkpoint. No saved result here establishes
the current four-class all-DT-MSE experiment. The four-class class-specific
values are those in the original JSONs; their association with the corresponding
thesis table entries is unresolved. Some paper summaries and the qualitative
figures also lack an exact checkpoint binding, as noted in the index.
