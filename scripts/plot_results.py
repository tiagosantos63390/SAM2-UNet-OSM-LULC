"""Plot aggregate_metrics.py summaries, including validation sweeps."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--metric', choices=['miou', 'f1', 'bmiou', 'road_iou', 'building_iou'], default='miou')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = json.loads(args.summary.read_text(encoding='utf-8'))
    groups = [group for group in groups if args.metric in group['scores']]
    if not groups:
        raise ValueError(f'No {args.metric} values in summary')
    scopes = {(g['provenance'], g['stage'], g['ddof']) for g in groups}
    if len(scopes) != 1:
        raise ValueError('Plot one provenance, evaluation stage and SD convention at a time')
    labels = []
    for group in groups:
        family = group['family'].removeprefix('sam2unet-train-kaggle-')
        count = group['scores'][args.metric]['n']
        labels.append(f"{family} / {group['configuration']} (n={count})")
    scores = [g['scores'][args.metric] for g in groups]
    fig, ax = plt.subplots(figsize=(11, max(4, .5 * len(groups))))
    for i, score in enumerate(scores):
        ax.barh(i, score['mean'] * 100, xerr=score['std'] * 100 if score['std'] is not None else None,
                color='#2f6f4f', capsize=3)
    ax.set_yticks(range(len(groups)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(f'{args.metric} (%)')
    provenance, stage, ddof = next(iter(scopes))
    ax.set_title(f'{provenance}, {stage}; SD ddof={ddof}')
    ax.grid(axis='x', alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
