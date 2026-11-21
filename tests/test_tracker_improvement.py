import unittest

import numpy as np

from scripts.tracker_improvement import low_confidence_cost, recovery_risk_cost, rerank_cost


class RerankCostTests(unittest.TestCase):
    def test_risk_gate_can_admit_shifted_box_with_strong_appearance(self):
        geometry = np.array([[0.56, 0.89]])
        appearance = np.array([[0.04, 0.02]])
        revised = recovery_risk_cost(geometry, appearance, [0.4, 0.55], [2], 0.37)
        self.assertLess(revised[0, 0], 0.5)
        self.assertEqual(revised[0, 1], 1.0)

    def test_risk_gate_preserves_active_rows_and_defers_ambiguous_pairs(self):
        geometry = np.array([[0.39, 0.9], [0.38, 0.49], [0.80, 0.48]])
        appearance = np.array([[0.12, 0.75], [0.04, 0.04], [0.4, 0.05]])
        revised = recovery_risk_cost(geometry, appearance, [0.31, 0.45], [0, 2, 2], 0.43)
        np.testing.assert_array_equal(revised[0], geometry[0])
        np.testing.assert_array_equal(revised[1:], np.ones((2, 2)))

    def test_reweights_a_contested_group_without_admitting_illegal_edges(self):
        baseline = np.array([[0.42, 0.40, 0.84], [0.41, 0.43, 0.95]])
        geom = np.array([[0.43, 0.60, 0.89], [0.62, 0.44, 0.95]])
        appearance = np.array([[0.44, 0.40, 0.31], [0.41, 0.43, 0.33]])
        revised = rerank_cost(baseline, geom, appearance, 0.8, 0.20, 0.5)
        np.testing.assert_array_equal(revised[:, 2], baseline[:, 2])
        self.assertLess(revised[0, 0] + revised[1, 1], revised[0, 1] + revised[1, 0])
        np.testing.assert_array_equal(baseline, [[0.42, 0.40, 0.84], [0.41, 0.43, 0.95]])

    def test_preserves_isolated_or_clear_decisions(self):
        baseline = np.array([[0.10, 0.50], [0.43, 0.12]])
        changed = rerank_cost(baseline, np.ones((2, 2)), np.ones((2, 2)), 0.8, 0.1, 0.5)
        np.testing.assert_array_equal(changed, baseline)

    def test_preserves_empty_costs(self):
        empty = np.empty((0, 2))
        self.assertEqual(rerank_cost(empty, empty, empty, 0.8, 0.1, 0.5).shape, (0, 2))

    def test_recovery_cannot_take_a_detection_available_to_an_active_track(self):
        geometry = np.array([[0.38, 0.67], [0.33, 0.36]])
        appearance = np.array([[0.15, 0.75], [0.04, 0.10]])
        cost = low_confidence_cost(geometry, appearance, [0.31, 0.42], [False, True])
        self.assertLess(cost[0, 0], 0.5)
        self.assertGreater(cost[1, 0], 0.5)
        self.assertLess(cost[1, 1], 0.5)

    def test_recovery_rejects_a_close_identity_competitor(self):
        geometry = np.array([[0.37], [0.39]])
        appearance = np.array([[0.08], [0.09]])
        cost = low_confidence_cost(geometry, appearance, [0.45], [True, True])
        self.assertTrue(np.all(cost > 0.5))

    def test_recovery_only_preserves_the_active_tracker_costs(self):
        geometry = np.array([[0.38, 0.56], [0.33, 0.36]])
        appearance = np.array([[0.15, 0.09], [0.04, 0.10]])
        cost = low_confidence_cost(geometry, appearance, [0.31, 0.42], [False, True], adjust_active=False)
        np.testing.assert_array_equal(cost[0], geometry[0])
        self.assertGreater(cost[1, 0], 0.5)
        self.assertLess(cost[1, 1], 0.5)
