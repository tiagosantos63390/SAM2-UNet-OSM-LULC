import json
import numpy as np
from pathlib import Path

# Path to the directory where all JSON files from Seed 42, 43, 44 are placed
HISTORY_DIR = Path('history')

SEEDS = [42, 43, 44]

# The base names corresponding to the 3 rows of Table 1
RUN_CONFIGS = [
    ("Baseline SAM2-UNet", "baseline"),
    ("OSM Auxiliary DT (Roads Only)", "roads_only_L0p5"),
    ("OSM Auxiliary DT (Roads+Buildings)", "roads_bldg_L0p25")
]

print("==========================================================================================")
print("COMPUTED TABLE 1 VALUES (TEST SET: MEAN ± STD DEV)")
print("==========================================================================================\n")

for label, base_name in RUN_CONFIGS:
    mious, f1s, bmious, road_ious = [], [], [], []
    
    for seed in SEEDS:
        json_file = HISTORY_DIR / f"{base_name}_seed{seed}_metrics.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                test_data = data['test']
                mious.append(test_data['miou'])
                f1s.append(test_data['f1'])
                bmious.append(test_data['bmiou'])
                road_ious.append(test_data['road_iou'])
        else:
            print(f"Warning: Could not find {json_file}")
            
    if len(mious) > 0:
        m_miou, s_miou = np.mean(mious), np.std(mious)
        m_f1, s_f1 = np.mean(f1s), np.std(f1s)
        m_bmiou, s_bmiou = np.mean(bmious), np.std(bmious)
        m_road, s_road = np.mean(road_ious), np.std(road_ious)
        
        # Formats directly into LaTeX table row format
        print(f"%-35s & {m_miou:.4f} $\\pm$ {s_miou:.4f} & {m_f1:.4f} $\\pm$ {s_f1:.4f} & {m_bmiou:.4f} $\\pm$ {s_bmiou:.4f} & {m_road:.4f} $\\pm$ {s_road:.4f} \\\\" % label)