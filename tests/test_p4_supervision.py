import unittest

from scripts.p4_supervision import candidate_features, current_quality, fit_immediate_rewards, matched_people


class CurrentFrameSupervisionTest(unittest.TestCase):
    def test_identity_labels_reward_the_correct_person(self):
        legal = {1: {3: 0.8}, 2: {3: 0.9}}
        anchors = {1: 101, 2: 202}
        persons = {3: 101}
        self.assertEqual(current_quality((1, 2), (3, None), legal, anchors, persons), 1)
        self.assertEqual(current_quality((1, 2), (None, 3), legal, anchors, persons), -2)

    def test_missed_person_outside_candidate_gate_is_not_penalized(self):
        self.assertEqual(current_quality((1,), (None,), {1: {}}, {1: 2}, {3: 2}), 0)

    def test_detection_labels_are_one_to_one(self):
        boxes = [(0, 0, 10, 10, 0.8), (0, 0, 10, 10, 0.7)]
        assignments = matched_people(boxes, {99: (0, 0, 10, 10)})
        self.assertEqual(len(assignments), 1)
        self.assertEqual(list(assignments.values()), [99])

    def test_action_features_include_misses_without_accessing_labels(self):
        x = candidate_features((1, 2), (3, None), {1: {3: 0.8}, 2: {}}, {3: 0.5})
        self.assertEqual(x, (0.5, 0.4, 0.25, 0.2))

    def test_immediate_fit_preserves_reward_magnitude(self):
        features = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
        weights = fit_immediate_rewards(features, [0.5, -1.0, 0.0])
        self.assertAlmostEqual(weights[0], 0.5 / 1.03)
        self.assertAlmostEqual(weights[1], -1.0 / 1.03)
