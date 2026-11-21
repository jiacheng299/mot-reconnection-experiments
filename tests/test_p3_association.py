import unittest

from scripts.p3_association import competition_groups, feasible_actions, select_linear_actions


class AssociationFeasibilityTest(unittest.TestCase):
    def test_two_tracks_share_one_detection(self):
        scores = {101: {7: 0.9}, 102: {7: 0.8}}
        self.assertEqual(competition_groups(scores), [(101, 102)])
        self.assertCountEqual(feasible_actions((101, 102), scores), [
            ((7, None), 0.9), ((None, 7), 0.8), ((None, None), 0.0),
        ])
        self.assertEqual(select_linear_actions(scores), {101: 7, 102: None})

    def test_competition_is_transitive_and_independent_groups_can_merge(self):
        scores = {4: {9: 0.6}, 3: {6: 0.9}, 2: {5: 0.6, 6: 0.7}, 1: {5: 0.8}, 5: {}}
        self.assertEqual(competition_groups(scores), [(1, 2, 3), (4,), (5,)])
        self.assertEqual(select_linear_actions(scores), {1: 5, 2: None, 3: 6, 4: 9, 5: None})

    def test_all_miss_action_is_available_even_when_every_track_has_candidates(self):
        scores = {1: {5: -0.1}, 2: {5: -0.3}}
        self.assertIn(((None, None), 0.0), feasible_actions((1, 2), scores))
        self.assertEqual(select_linear_actions(scores), {1: None, 2: None})

    def test_empty_track_set(self):
        self.assertEqual(competition_groups({}), [])
        self.assertEqual(select_linear_actions({}), {})

    def test_equal_scores_use_stable_id_order(self):
        self.assertEqual(select_linear_actions({2: {5: 0.8}, 1: {5: 0.8}}), {1: 5, 2: None})
