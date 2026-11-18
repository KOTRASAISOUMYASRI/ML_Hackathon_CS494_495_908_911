import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from route_recovery.classifier import score_candidates, model_paths, _load
from route_recovery.data_adapter import SharedRoadTrip, trip_files
from route_recovery.evaluation import evaluate_reference
from route_recovery.hmm import recover


class ClassifierInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name) / '00000001'
        self.base.with_suffix('.nodes').write_text('8 49\n8.001 49\n8.002 49\n')
        self.base.with_suffix('.arcs').write_text('0 1\n1 2\n')
        self.base.with_suffix('.track').write_text('8.0001 49 1\n8.0019 49 3\n')
        self.files = trip_files(self.directory.name, '00000001')
        self.trip = SharedRoadTrip(self.files)

    def test_existing_baseline_and_emission_normalization(self):
        scorer, probabilities = score_candidates(self.trip, 'decision_tree')
        self.assertEqual(len(probabilities), len(self.trip.features.candidates))
        for observation, candidates in zip(self.trip.track, self.trip.candidate_rows):
            self.assertAlmostEqual(float(np.exp(scorer(observation, candidates)).sum()), 1)

    def test_inference_ignores_reference(self):
        self.base.with_suffix('.route').write_text('INVALID REFERENCE')
        scorer, _ = score_candidates(self.trip, 'decision_tree')
        result = recover(self.trip, scorer=scorer)
        self.assertEqual(result['mode'], 'classifier_hmm')
        self.assertNotIn('target', self.trip.features.candidates)

    def test_candidate_misses_count_as_top1_errors(self):
        self.base.with_suffix('.route').write_text('0\n1\n')
        from data_loader import TripFiles
        reference_files = TripFiles(self.files.trip_id, self.files.track, self.files.nodes,
                                    self.files.arcs, self.base.with_suffix('.route'))
        scorer, p = score_candidates(self.trip, 'decision_tree')
        result = recover(self.trip, scorer=scorer)
        with patch('route_recovery.evaluation._align_reference', return_value=np.array([999, 999])):
            metrics = evaluate_reference(self.trip, result, reference_files, p)
        self.assertEqual(metrics['candidate_misses'], 2)
        self.assertEqual(metrics['hmm']['top1_accuracy'], 0)
        self.assertEqual(metrics['independent']['top1_accuracy'], 0)
        self.assertIn('macro_f1', metrics['candidate'])

    def test_candidate_order_mismatch_rejected(self):
        scorer, _ = score_candidates(self.trip, 'decision_tree')
        with self.assertRaises(ValueError):
            scorer(self.trip.track[0], [])

    def test_api_evaluation_is_optional_and_keeps_references_private(self):
        from fastapi.testclient import TestClient
        from route_recovery import main
        root = Path(self.directory.name)
        with patch.object(main, 'inventory', return_value={'00000001': self.files}):
            client = TestClient(main.app)
            self.base.with_suffix('.route').write_text('0\n1\n')
            from data_loader import TripFiles
            references = TripFiles(self.files.trip_id, self.files.track, self.files.nodes,
                                   self.files.arcs, self.base.with_suffix('.route'))
            with patch.object(main, 'inventory', return_value={'00000001': references}):
                r = client.post('/match', json={'trip_id':'00000001','model':'decision_tree'})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertNotIn('evaluation', r.json())
                evaluated = client.post('/visualize', json={'trip_id':'00000001','model':'decision_tree','evaluate':True})
                self.assertEqual(evaluated.status_code, 200, evaluated.text)
                self.assertIn('evaluation', evaluated.json()['result'])
                self.assertNotIn('reference_arc_id', evaluated.text)
                self.assertNotIn('reference_arcs', evaluated.text)
                rejected = client.post('/match', json={'trip_id':'00000001','model':'../../unknown'})
                self.assertEqual(rejected.status_code, 422)
                changed = client.post('/match', json={'trip_id':'00000001','model':'decision_tree','radius_m':100})
                self.assertEqual(changed.status_code, 422)
                self.assertEqual(client.get('/app.js').status_code, 200)

    def test_evaluator_inventory_uses_configured_input(self):
        from fastapi.testclient import TestClient
        from route_recovery import main
        root = Path(self.directory.name) / 'evaluator'
        trip = root / '00000001'
        trip.mkdir(parents=True)
        (trip / '00000001.track').write_text('8.0001 49 1\n')
        (trip / '00000001.nodes').write_text('8 49\n8.001 49\n')
        (trip / '00000001.arcs').write_text('0 1\n')
        with patch.object(main, 'EVALUATOR_INPUT', str(root)):
            client = TestClient(main.app)
            self.assertEqual(client.get('/trips').json()['evaluator'], ['00000001/00000001'])
        with patch.object(main, 'EVALUATOR_INPUT', None):
            client = TestClient(main.app)
            self.assertEqual(client.get('/trips').json()['evaluator'], [])

    def test_health_discovers_member3_artifacts_without_loading(self):
        from fastapi.testclient import TestClient
        from route_recovery.main import app
        with patch('route_recovery.classifier.Path.is_file', return_value=True), patch(
                'route_recovery.classifier.joblib.load') as load:
            available = TestClient(app).get('/health').json()['models']
            for name in ('random_forest', 'hist_gradient_boosting',
                         'random_forest_optimized', 'hist_gradient_boosting_optimized'):
                self.assertIn(name, available)
            load.assert_not_called()

    def test_lfs_pointer_has_actionable_error(self):
        artifact = Path(self.directory.name) / 'pointer.joblib'
        artifact.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:example\nsize 100\n')
        with self.assertRaisesRegex(ValueError, 'git lfs pull'):
            _load(str(artifact), artifact.stat().st_mtime_ns)

    def test_selected_hgb_is_default_and_comparison_is_measured(self):
        from fastapi.testclient import TestClient
        from route_recovery import main
        from route_recovery.selection import SELECTED_MODEL
        client = TestClient(main.app)
        self.assertEqual(main.MatchRequest(trip_id='example').model, SELECTED_MODEL)
        health = client.get('/health').json()
        self.assertEqual(health['default_model'], SELECTED_MODEL)
        self.assertTrue(health['final_model_selected'])
        comparison = client.get('/selected-model').json()
        self.assertTrue(comparison['report_available'], comparison)
        self.assertEqual(comparison['summary']['observations'], 19160)
        self.assertLess(comparison['changes']['top1_percentage_points'], 0)
        self.assertGreater(comparison['changes']['continuity_percentage_points'], 0)
        self.assertNotIn('trips', comparison)
        with patch.object(main, 'inventory', return_value={'00000001': self.files}):
            response = client.post('/match', json={'trip_id':'00000001'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['model'], SELECTED_MODEL)
            self.assertEqual(response.json()['mode'], 'classifier_hmm')

    def test_stale_selected_comparison_is_not_displayed(self):
        from route_recovery.selection import selected_comparison, SELECTED_MODEL
        artifact = Path(self.directory.name) / 'changed.joblib'
        artifact.write_bytes(b'new artifact not yet evaluated')
        with patch('route_recovery.selection.model_paths', return_value={SELECTED_MODEL:artifact}):
            result = selected_comparison()
        self.assertFalse(result['report_available'])
        self.assertNotIn('summary', result)
        self.assertIn('stale', result['message'])
