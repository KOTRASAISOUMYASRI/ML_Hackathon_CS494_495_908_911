import itertools
import tempfile
import unittest
from pathlib import Path
from route_recovery.hmm import RoadTrip, recover
from route_recovery.viterbi import NoValidPath, decode


class DecoderTests(unittest.TestCase):
    def test_exhaustive_oracle(self):
        emissions = [[-1, -2], [-3, -.1, -2], [-.2, -1]]
        transitions = [[[-1, -9, -2], [-2, -1, -3]], [[-1, -4], [-2, -1], [-1, -2]]]
        def score(p):
            return sum(emissions[t][j] for t, j in enumerate(p)) + sum(transitions[t][p[t]][p[t+1]] for t in range(2))
        expected = max(itertools.product(range(2), range(3), range(2)), key=score)
        path, value = decode(emissions, transitions)
        self.assertEqual(path, list(expected))
        self.assertAlmostEqual(value, score(expected))

    def test_invalid_and_impossible(self):
        with self.assertRaises(NoValidPath):
            decode([[0], [0]], [[[float('-inf')]]])
        with self.assertRaises(ValueError):
            decode([[float('nan')]], [])
        with self.assertRaises(ValueError):
            decode([[0], [0]], [[[0, 0]]])
        self.assertEqual(decode([], []), ([], 0))

    def test_long_sequence(self):
        path, score = decode([[-1000]] * 2000, [[[0]]] * 1999)
        self.assertEqual(len(path), 2000)
        self.assertEqual(score, -2000000)

    def test_trip_and_gap_without_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / '00000001'
            base.with_suffix('.nodes').write_text('8 49\n8.001 49\n8.002 49\n')
            base.with_suffix('.arcs').write_text('0 1\n1 2\n')
            base.with_suffix('.track').write_text('8.0001 49 1\n8.0011 49 2\n8.0015 49 100\n')
            result = recover(RoadTrip(folder, '00000001'), max_speed_mps=200)
            graph = RoadTrip(folder, '00000001')
            self.assertEqual(graph.shortest(0, 2, 10)[0], float('inf'))
            self.assertTrue(graph.shortest(0, 2, 200)[0] < 200)
            self.assertEqual(result['matched_observations'], 3)
            self.assertEqual(len(result['route_sections']), 2)
            self.assertEqual(result['breaks'][0]['reason'], 'timestamp_gap')
            self.assertFalse(base.with_suffix('.route').exists())
