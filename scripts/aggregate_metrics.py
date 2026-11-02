"""Aggregate saved runs by configuration, split and historical/current provenance."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ('miou', 'f1', 'bmiou', 'road_iou', 'building_iou')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def indexed_runs(index, stage):
    for entry in read_json(index)['runs']:
        path = REPO_ROOT / entry['metrics']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['metrics_sha256']:
            raise ValueError(f'Historical artifact hash mismatch: {path}')
        data = read_json(path)
        if stage == 'best-val':
            if 'validation_history' in entry:
                history = read_json(REPO_ROOT / entry['validation_history'])['history']
                values = max(history, key=lambda row: row['miou'])
            elif 'validation_summary' in entry:
                values = entry['validation_summary']
            else:
                values = {'miou': data['best_miou'], 'epoch': data['best_epoch']}
        else:
            values = data.get(stage)
        if values:
            yield {**entry, 'stage': stage, 'values': values}


def current_runs(history_dir, stage):
    for path in sorted(Path(history_dir).glob('*_metrics.json')):
        data = read_json(path)
        if 'configuration' not in data or 'seed' not in data:
            raise ValueError(f'{path} has no current configuration/seed; use the historical index')
        config = data['configuration']
        identity = {key: value for key, value in config.items()
                    if key not in ('seeds', 'work_dir', 'experiments')}
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        values = data.get('val' if stage == 'best-val' else stage)
        if values:
            yield {'provenance': 'current', 'family': 'public-modules',
                   'configuration': config['name'], 'configuration_id': signature,
                   'split_identity': data.get('split_sha256', {}), 'seed': data['seed'],
                   'metrics': str(path), 'stage': stage, 'values': values}


def aggregate(runs, ddof=0):
    if ddof not in (0, 1):
        raise ValueError('ddof must be 0 (population) or 1 (sample)')
    groups = defaultdict(list)
    for run in runs:
        key = (run['provenance'], run['family'], run['configuration'],
               run.get('configuration_id', ''), json.dumps(run['split_identity'], sort_keys=True), run['stage'])
        groups[key].append(run)
    output = []
    for key, members in sorted(groups.items()):
        seeds = [run['seed'] for run in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'Duplicate seed within configuration: {key}')
        scores = {}
        for metric in METRICS:
            values = []
            for run in members:
                value = run['values'].get(metric)
                if metric == 'building_iou' and value is None:
                    value = run['values'].get('bldg_iou')
                    if value is None and 'per_class_iou' in run['values']:
                        value = run['values']['per_class_iou'][8]
                if value is not None:
                    values.append(value)
            if values:
                if ddof == 0:
                    std = statistics.pstdev(values)
                elif len(values) > 1:
                    std = statistics.stdev(values)
                else:
                    std = None
                scores[metric] = {'mean': statistics.mean(values), 'std': std, 'n': len(values)}
        first = members[0]
        summary = {field: first[field] for field in
                   ('provenance', 'family', 'configuration', 'split_identity', 'stage')}
        summary.update(configuration_id=first.get('configuration_id', ''), ddof=ddof,
                       seeds=seeds, scores=scores, runs=members)
        output.append(summary)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--index', type=Path, default=None)
    source.add_argument('--history-dir', type=Path, default=None)
    parser.add_argument('--stage', choices=['test', 'val', 'best-val'], default='test')
    parser.add_argument('--ddof', choices=[0, 1], type=int, default=0, help='0: population SD; 1: sample SD')
    parser.add_argument('--family', default=None, help='Exact historical notebook family')
    parser.add_argument('--output', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.history_dir:
        runs = current_runs(args.history_dir, args.stage)
    else:
        runs = indexed_runs(args.index or REPO_ROOT / 'results/index.json', args.stage)
    groups = aggregate((run for run in runs if args.family is None or run['family'] == args.family), args.ddof)
    if not groups:
        raise ValueError('No saved results match this source, stage and family')
    print(f'{args.stage}; SD ddof={args.ddof}; scores in [0, 1]')
    for group in groups:
        scores = []
        for metric, score in group['scores'].items():
            std = 'undefined' if score['std'] is None else f"{score['std']:.4f}"
            scores.append(f"{metric}={score['mean']:.4f} +/- {std} (n={score['n']})")
        print(f"{group['family']} / {group['configuration']} seeds={group['seeds']}: " + ', '.join(scores))
    if args.output:
        args.output.write_text(json.dumps(groups, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
