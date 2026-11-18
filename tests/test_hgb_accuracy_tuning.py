import unittest
import numpy as np
import pandas as pd
from experiments.tune_hgb_accuracy import measure


class AccuracyObjectiveTests(unittest.TestCase):
    def test_trip_ids_separate_observations_and_misses_count_as_errors(self):
        table=pd.DataFrame({'trip_id':['a','a','b'],
                            'observation_id':[0,0,0],'target':[0,1,0]})
        result=measure(table,np.array([.1,.8,.9]),observation_count=3)
        self.assertEqual(result['correct'],1)
        # Includes an all-negative group and an observation with no candidates.
        self.assertAlmostEqual(result['top1'],1/3)

    def test_top1_ranks_candidates_without_binary_threshold(self):
        table=pd.DataFrame({'trip_id':['a','a'],
                            'observation_id':[0,0],'target':[0,1]})
        result=measure(table,np.array([.1,.2]),observation_count=1)
        self.assertEqual(result['top1'],1)
        self.assertEqual(result['candidate_accuracy'],.5)
