import json
from pathlib import Path
import tempfile
import unittest

from scripts.aggregate_metrics import aggregate, current_runs, indexed_runs

INDEX = Path(__file__).resolve().parents[1] / 'results' / 'index.json'


class ResultsTests(unittest.TestCase):
    def test_saved_results_and_validation_selection(self):
        runs = list(indexed_runs(INDEX, 'test'))
        self.assertEqual(len(runs), 37)
        groups = aggregate(runs)
        baseline = next(g for g in groups if g['configuration'] == 'baseline')
        self.assertAlmostEqual(baseline['scores']['miou']['mean'], .6544853514697031)
        self.assertAlmostEqual(baseline['scores']['miou']['std'], .010126165862531296)
        validation = list(indexed_runs(INDEX, 'best-val'))
        d6 = next(r for r in validation if r['configuration'] == 'tiny_dt_lambda_0p5_d6p0')
        self.assertEqual(d6['values']['epoch'], 49)
        self.assertAlmostEqual(d6['values']['miou'], .6776982432239128)

    def test_separate_families_and_duplicate_seeds(self):
        runs = list(indexed_runs(INDEX, 'test'))
        combined = [r for r in runs if r['configuration'] == 'roads_bldg_L0p25']
        self.assertEqual(len(aggregate(combined)), 2)
        with self.assertRaises(ValueError):
            aggregate([runs[0], runs[0]])
        self.assertIsNone(aggregate(runs[:1], ddof=1)[0]['scores']['miou']['std'])

    def test_validation_only_runs_and_different_distances(self):
        with tempfile.TemporaryDirectory() as directory:
            for distance in (3, 6, 10):
                path = Path(directory) / f'd{distance}_metrics.json'
                path.write_text(json.dumps({
                    'configuration': {'name': 'roads', 'dt_max_distance': distance},
                    'seed': 42, 'val': {'miou': .6}, 'test': None,
                }), encoding='utf-8')
            self.assertEqual(list(current_runs(directory, 'test')), [])
            groups = aggregate(current_runs(directory, 'best-val'))
            self.assertEqual(len(groups), 3)
            self.assertTrue(all(g['scores']['miou']['mean'] == .6 for g in groups))


if __name__ == '__main__':
    unittest.main()
