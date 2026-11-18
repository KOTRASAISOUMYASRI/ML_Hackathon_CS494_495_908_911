import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from route_recovery.hmm import Candidate, RoadTrip, minimum_connection, recover
from route_recovery.data_adapter import SharedRoadTrip, trip_files


class RoadEdgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def graph(self, arcs='0 1\n1 2\n', track='8.0001 49 1\n8.0019 49 2\n'):
        base = Path(self.temp.name) / '00000001'
        base.with_suffix('.nodes').write_text('8 49\n8.001 49\n8.002 49\n8.01 49\n8.011 49\n')
        base.with_suffix('.arcs').write_text(arcs)
        base.with_suffix('.track').write_text(track)
        return RoadTrip(self.temp.name, '00000001')

    def test_backwards_requires_directed_cycle(self):
        trip = self.graph()
        a, b = Candidate(0, .8, 0, (0, 0)), Candidate(0, .2, 0, (0, 0))
        self.assertEqual(trip.connect(a, b)[0], float('inf'))
        trip = self.graph('0 1\n1 0\n')
        distance, path = trip.connect(a, b)
        self.assertEqual(path, (1,))
        self.assertAlmostEqual(distance, 1.4 * trip.lengths[0])

    def test_missing_candidates_split_without_connector(self):
        trip = self.graph(track='8.0001 49 1\n8.0002 49 2\n8.0003 49 3\n')
        row = trip.candidates(trip.track[0])[:1]
        trip.candidate_rows = [row, [], row]
        result = recover(trip)
        self.assertEqual(result['arc_ids'][1], None)
        self.assertEqual(len(result['route_sections']), 2)
        self.assertEqual(result['breaks'][0]['reason'], 'no_usable_candidates')

    def test_geometry_includes_intermediate_road_nodes(self):
        trip = self.graph()
        trip.candidate_rows = [trip.candidates(trip.track[0], 1), trip.candidates(trip.track[1], 1)]
        result = recover(trip, max_speed_mps=200)
        self.assertEqual(len(result['route_sections']), 1)
        self.assertIn([49.0, 8.001], result['route_sections'][0])

    def test_disconnected_components_diagnostic(self):
        trip = self.graph('0 1\n3 4\n', '8.0001 49 1\n8.0101 49 2\n')
        trip.candidate_rows = [trip.candidates(o, 1) for o in trip.track]
        result = recover(trip, diagnose=True)
        self.assertEqual(result['breaks'][0]['cause'], 'disconnected_components')
        self.assertEqual(len(result['route_sections']), 2)

    def test_speed_limit_diagnostic(self):
        trip = self.graph()
        trip.candidate_rows = [trip.candidates(o, 1) for o in trip.track]
        result = recover(trip, diagnose=True)
        detail = result['breaks'][0]
        self.assertEqual(detail['cause'], 'speed_limit')
        self.assertGreater(detail['minimum_network_m'], detail['allowed_network_m'])
        self.assertAlmostEqual(detail['minimum_network_m'], minimum_connection(trip, *trip.candidate_rows))

    def test_direction_diagnostic(self):
        trip = self.graph(track='8.0008 49 1\n8.0002 49 2\n')
        trip.candidate_rows = [trip.candidates(o, 1) for o in trip.track]
        self.assertEqual(recover(trip, diagnose=True)['breaks'][0]['cause'], 'road_direction')

    def test_off_network_fallback_matches_shared_pipeline(self):
        self.graph(track='8.0005 49.01 1\n')
        trip = SharedRoadTrip(trip_files(self.temp.name, '00000001'))
        self.assertEqual(len(trip.candidate_rows[0]), 1)
        self.assertGreater(trip.candidate_rows[0][0].distance_m, 50)

    def test_visualize_and_cross_origin_preflight(self):
        from fastapi.testclient import TestClient
        from route_recovery import main, ui_server
        root = Path(self.temp.name)
        folder = root / 'student_test'
        folder.mkdir()
        base = folder / '00000001'
        base.with_suffix('.nodes').write_text('8 49\n8.001 49\n')
        base.with_suffix('.arcs').write_text('0 1\n')
        base.with_suffix('.track').write_text('8.0001 49 1\n')
        with patch.object(main, 'DATA_ROOT', root):
            client = TestClient(main.app)
            response = client.options('/visualize', headers={'Origin': 'http://localhost:8080',
                'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['access-control-allow-origin'], 'http://localhost:8080')
            response = client.post('/visualize', json={'trip_id': '00000001'}, headers={'Origin': 'http://localhost:8080'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn('Road network', response.json()['map_html'])
            self.assertEqual(response.json()['result']['unmatched_observations'], 0)
            self.assertIn('Backend API address', TestClient(ui_server.app).get('/').text)
