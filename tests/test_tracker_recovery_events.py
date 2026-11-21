import unittest

from scripts.tracker_recovery_events import recovery_events, continuation


class RecoveryEventsTests(unittest.TestCase):
    def test_uses_presence_and_detector_scores_to_define_shared_events(self):
        truth = {frame: {8: None} for frame in range(1, 7)}
        truth[3][9] = None
        scores = {1: {8: .92}, 2: {}, 3: {8: .32, 9: .21}, 4: {8: .85}, 5: {}, 6: {8: .92}}
        self.assertEqual([(event['before'], event['return'], event['group']) for event in recovery_events(truth, scores)],
                         [(1, 3, 'gap_low'), (4, 6, 'gap_high')])

    def test_counts_missing_anchor_and_wrong_identity_separately(self):
        event = {'identity': 8, 'before': 5, 'return': 7}
        self.assertEqual(continuation(event, {5:{8: 9}, 7:{8: 9}}), 'correct')
        self.assertEqual(continuation(event, {5:{8: 9}, 7:{8: 6}}), 'switched')
        self.assertEqual(continuation(event, {5:{8: 9}, 7:{}}), 'missed')
        self.assertEqual(continuation(event, {5:{}, 7:{8: 6}}), 'no_anchor')
