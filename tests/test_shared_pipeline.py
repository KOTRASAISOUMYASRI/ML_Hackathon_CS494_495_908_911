"""Integration tests against the exact feature contract used for training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from route_recovery.data_adapter import SharedRoadTrip, trip_files
from route_recovery.hmm import recover


class SharedPipelineTests(unittest.TestCase):
    def test_candidates_ids_and_reference_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / '00000001'
            base.with_suffix('.nodes').write_text('8 49\n8.001 49\n8.002 49\n')
            base.with_suffix('.arcs').write_text('0 1\n1 2\n')
            base.with_suffix('.track').write_text('8.0001 49 1\nbad row\n8.0011 49 3\n')
            base.with_suffix('.route').write_text('INVALID REFERENCE MUST NOT BE READ')
            trip = SharedRoadTrip(trip_files(folder, '00000001'))
            self.assertEqual(trip.observation_ids, [0, 2])
            self.assertNotIn('target', trip.features.candidates)
            for observation_id, candidates in zip(trip.observation_ids, trip.candidate_rows):
                expected = trip.features.candidates.query('observation_id == @observation_id')
                self.assertEqual([c.arc_id for c in candidates], expected.candidate_arc_id.tolist())
                self.assertEqual([c.fraction for c in candidates], expected.along_segment_fraction.tolist())
            result = recover(trip)
            self.assertEqual(result['observation_ids'], [0, 2])
            self.assertEqual(result['breaks'][0]['reason'], 'removed_observations')
            self.assertEqual(len(result['route_sections']), 2)

    def test_api_uses_inventory_ids(self):
        from fastapi.testclient import TestClient
        from route_recovery import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'student_test' / '00000001' / '00000001'
            base.parent.mkdir(parents=True)
            base.with_suffix('.nodes').write_text('8 49\n8.001 49\n')
            base.with_suffix('.arcs').write_text('0 1\n')
            base.with_suffix('.track').write_text('8.0001 49 1\n8.0002 49 2\n')
            with patch.object(main, 'DATA_ROOT', root):
                client = TestClient(main.app)
                ids = client.get('/trips').json()['student_test']
                self.assertEqual(ids, ['00000001/00000001'])
                response = client.post('/match', json={'trip_id': ids[0]})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['observation_ids'], [0, 1])
                self.assertEqual(client.post('/match', json={'trip_id': '../outside'}).status_code, 404)
