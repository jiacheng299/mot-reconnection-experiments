import unittest

import numpy as np

from reconnect_study.tracker import run_bytetrack


BOX = (10.0, 20.0, 30.0, 60.0, 0.95)


class ByteTrackReplayTest(unittest.TestCase):
    def test_default_policy_keeps_the_id_and_records_both_observations(self):
        result = run_bytetrack({1: [BOX], 3: [BOX]}, 3, sequence="demo")

        self.assertEqual([(row.frame, row.track_id) for row in result.tracks], [(1, 1), (3, 1)])
        self.assertEqual(len(result.decisions), 1)
        decision = result.decisions[0]
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.proposal.sequence, "demo")
        self.assertEqual(decision.proposal.previous_frame, 1)
        self.assertEqual(decision.proposal.frame, 3)
        self.assertEqual(decision.proposal.previous_box, BOX[:4])
        self.assertEqual(decision.proposal.candidate_box, BOX[:4])
        self.assertEqual(decision.proposal.features["gap_frames"], 1.0)
        self.assertEqual(decision.proposal.features["iou"], 1.0)

    def test_rejection_changes_future_track_state_and_allows_new_id(self):
        shifted = (15.0, 20.0, 35.0, 60.0, 0.95)
        detections = {1: [BOX], 3: [shifted], 4: [shifted], 5: [shifted]}
        accepted = run_bytetrack(detections, 5)
        rejected = run_bytetrack(detections, 5, policy=lambda _: False)

        self.assertEqual([(row.frame, row.track_id) for row in accepted.tracks], [(1, 1), (3, 1), (4, 1), (5, 1)])
        self.assertEqual([(row.frame, row.track_id) for row in rejected.tracks], [(1, 1), (4, 2), (5, 2)])
        self.assertGreaterEqual(len(rejected.decisions), 1)
        self.assertTrue(all(not decision.accepted for decision in rejected.decisions))

    def test_normal_low_confidence_continuation_does_not_call_reconnect_policy(self):
        def fail_on_reconnect(_):
            self.fail("A continuous track must not require authorization for an ordinary update")

        low_confidence = (*BOX[:4], 0.2)
        result = run_bytetrack({1: [BOX], 2: [low_confidence]}, 2, policy=fail_on_reconnect)

        self.assertEqual([(row.frame, row.track_id) for row in result.tracks], [(1, 1), (2, 1)])
        self.assertEqual(result.decisions, ())

    def test_proposal_preserves_raw_history_through_an_ordinary_update(self):
        moved = (12.0, 20.0, 32.0, 60.0, 0.92)
        result = run_bytetrack({1: [BOX], 2: [moved], 4: [moved]}, 4)

        observation = result.tracks[1]
        proposal = result.decisions[0].proposal
        self.assertNotEqual(observation.box, moved[:4])
        self.assertEqual(proposal.previous_box, moved[:4])
        self.assertEqual(proposal.previous_frame, 2)
        self.assertEqual(proposal.features["previous_score"], moved[4])


class BoTSortReplayTest(unittest.TestCase):
    def test_cached_camera_motion_affects_closed_loop_reconnect(self):
        from reconnect_study.tracker import run_botsort

        shifted = (90.0, 20.0, 110.0, 60.0, 0.95)
        boxes = {1: [BOX], 3: [shifted]}
        stationary = run_botsort(boxes, 3)
        motion = {2: np.array([[1.0, 0.0, 80.0], [0.0, 1.0, 0.0]])}
        moving = run_botsort(boxes, 3, camera_warps_by_frame=motion)
        denied = run_botsort(boxes, 3, camera_warps_by_frame=motion, policy=lambda _: False)

        self.assertEqual(stationary.decisions, ())
        self.assertEqual([(row.frame, row.track_id) for row in moving.tracks], [(1, 1), (3, 1)])
        self.assertEqual(moving.decisions[0].proposal.predicted_box, shifted[:4])
        self.assertTrue(moving.decisions[0].accepted)
        self.assertFalse(denied.decisions[0].accepted)
        self.assertEqual([row.frame for row in denied.tracks], [1])

    def test_cached_appearance_scores_align_to_original_detection_rows(self):
        from reconnect_study.tracker import run_botsort

        low = (*BOX[:4], 0.05)
        nearby = (12.0, 20.0, 32.0, 60.0, 0.9)
        boxes = {1: [BOX], 3: [low, BOX, nearby]}
        features = {1: [[1.0, 0.0]], 3: [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]}

        geometry = run_botsort(boxes, 3)
        appearance = run_botsort(boxes, 3, appearance_by_frame=features)

        self.assertEqual(geometry.decisions[0].proposal.candidate_index, 1)
        self.assertEqual(appearance.decisions[0].proposal.candidate_index, 2)
        self.assertEqual(appearance.decisions[0].proposal.candidate_box, nearby[:4])


if __name__ == "__main__":
    unittest.main()
