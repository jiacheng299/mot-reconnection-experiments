import unittest
from unittest.mock import patch

from scripts.p3_c1 import choose_group_action


class BoundedGroupSearchTest(unittest.TestCase):
    def test_joint_choice_uses_group_total(self):
        scores = {1: {4: 0.9, 5: 0.8}, 2: {4: 0.7}}
        selected, candidates, fallback = choose_group_action((1, 2), scores, 16)
        self.assertEqual(selected, (5, 4))
        self.assertEqual(len(candidates), 5)
        self.assertFalse(fallback)

    def test_large_group_has_feasible_unique_candidates_with_exact_linear_seed(self):
        group = tuple(range(8))
        scores = {track: {det: 1.0 if det == track else 0.75 for det in group} for track in group}
        selected, candidates, fallback = choose_group_action(group, scores, 4)
        self.assertEqual(selected, group)
        self.assertLessEqual(len(candidates), 4)
        self.assertEqual(len(set(candidates)), len(candidates))
        self.assertFalse(fallback)
        for action in candidates:
            used = [det for det in action if det is not None]
            self.assertEqual(len(set(used)), len(used))

    def test_solver_error_has_a_legal_all_miss_fallback(self):
        scores = {1: {4: 0.9, 5: 0.8}, 2: {4: 0.7}}
        with patch('scripts.p3_c1.linear_sum_assignment', side_effect=ValueError('forced test error')):
            selected, candidates, fallback = choose_group_action((1, 2), scores, 1)
        self.assertEqual(selected, (None, None))
        self.assertEqual(candidates, [(None, None)])
        self.assertTrue(fallback)
