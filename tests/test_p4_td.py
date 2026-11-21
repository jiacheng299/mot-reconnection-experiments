import unittest

import numpy as np

from scripts.p4_td import next_frame_features, td_targets
from scripts.p4_joint import JointValue, encode_group, pack_candidates


class TemporalDifferenceTest(unittest.TestCase):
    def test_next_value_sums_new_groups_and_uses_separate_search_weights(self):
        # Two next-frame groups. The search and target weights prefer different candidates.
        groups = [
            [(2.0, 0.0), (0.0, 3.0)],
            [(1.0, 0.0), (0.0, 1.0)],
        ]
        feature_sum = next_frame_features(groups, np.array([1.0, 0.0]), 2)
        np.testing.assert_array_equal(feature_sum, [3.0, 0.0])
        rewards = np.array([0.5, -0.25])
        successors = np.array([feature_sum, [0.0, 0.0]])
        np.testing.assert_allclose(td_targets(rewards, successors, np.array([2.0, 10.0]), 0.9),
                                   [5.9, -0.25])

    def test_zero_gamma_recovers_immediate_targets(self):
        result = td_targets(np.array([-1.0, 0.5]), np.array([[100.0], [0.0]]),
                            np.array([3.0]), 0.0)
        np.testing.assert_array_equal(result, [-1.0, 0.5])

    def test_joint_score_backpropagates_through_each_component(self):
        legal = {1: {1: 0.8, 2: 0.65}, 2: {1: 0.7}}
        scores = {1: 0.9, 2: 0.95}
        state = {1: (1, 0, 1), 2: (1, 0, 2)}
        matched = encode_group((1, 2), (1, None), legal, scores, state)
        missed = encode_group((1, 2), (None, None), legal, scores, state)
        self.assertEqual(len(matched[1]), 1)
        self.assertEqual(len(matched[2]), 1)
        self.assertEqual(len(missed[2]), 2)
        model = JointValue()
        model(pack_candidates([matched, missed])).sum().backward()
        for component in (model.track, model.mix, model.pair, model.birth):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters()))
