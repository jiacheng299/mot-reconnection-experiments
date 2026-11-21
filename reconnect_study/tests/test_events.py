from types import SimpleNamespace as Row
import unittest

from reconnect_study.events import evaluate, evaluate_prepared, natural_gaps, pair, prepare


LEFT = (0, 0, 10, 10)
RIGHT = (20, 0, 30, 10)


class EventsTest(unittest.TestCase):
    def test_pair_prefers_matching_two_people_over_one_high_overlap(self):
        boxes = [(2, 0, 12, 10), (-3, 0, 7, 10)]
        people = {1: LEFT, 2: (6, 0, 16, 10)}
        self.assertEqual(pair(boxes, people, minimum_iou=0.4), {0: 2, 1: 1})

    def test_gap_requires_continuous_annotation_and_detector_absence(self):
        truth = {1: {4: LEFT, 8: RIGHT}, 2: {4: LEFT},
                 3: {4: LEFT, 8: RIGHT}, 4: {4: LEFT, 8: RIGHT}}
        detections = {1: [(*LEFT, .8), (*RIGHT, .8)], 3: [(*LEFT, .8), (*RIGHT, .9)],
                      4: [(*LEFT, .7)]}
        events = natural_gaps(detections, truth)
        self.assertEqual([(x.person, x.before, x.after) for x in events], [(4, 1, 3)])

    def test_missed_and_wrong_counts_are_distinct_from_track_switches(self):
        truth = {1: {4: LEFT, 8: RIGHT}, 2: {4: LEFT, 8: RIGHT},
                 3: {4: LEFT, 8: RIGHT}}
        detections = {1: [(*LEFT, .8), (*RIGHT, .8)], 3: [(*LEFT, .8), (*RIGHT, .9)]}
        tracks = [Row(frame=1, track_id=10, box=LEFT), Row(frame=1, track_id=11, box=RIGHT),
                  Row(frame=3, track_id=10, box=RIGHT)]
        proposals = [
            Row(proposal=Row(frame=3, track_id=10, previous_frame=1,
                             previous_box=LEFT, candidate_box=RIGHT, features={}), accepted=True),
            Row(proposal=Row(frame=3, track_id=11, previous_frame=1,
                             previous_box=RIGHT, candidate_box=RIGHT, features={}), accepted=False),
        ]
        outcome = evaluate(detections, truth, tracks, proposals)
        self.assertEqual(outcome.gaps, 2)
        self.assertEqual(outcome.correct, 0)
        self.assertEqual(outcome.recovery_outcomes, ("missed", "switched"))
        self.assertEqual((outcome.accepted, outcome.wrong, outcome.unverified_accepted), (1, 1, 0))
        self.assertEqual([x.correct for x in outcome.actions], [False, True])

    def test_unverifiable_acceptance_is_charged_to_risk(self):
        action = Row(proposal=Row(frame=2, track_id=10, previous_frame=1,
                                  previous_box=LEFT, candidate_box=RIGHT, features={}), accepted=True)
        result = evaluate({2: [(*RIGHT, .9)]}, {2: {8: RIGHT}}, [], [action])
        self.assertEqual((result.accepted, result.wrong, result.unverified_accepted), (1, 1, 1))

    def test_duplicate_box_sensitivity_is_opt_in(self):
        truth = {1: {4: LEFT}, 3: {4: LEFT}}
        evidence = prepare({1: [(*LEFT, .9)], 3: [(*LEFT, .1), (*LEFT, .9)]}, truth)
        track = Row(frame=1, track_id=10, box=LEFT)
        action = Row(proposal=Row(frame=3, track_id=10, previous_frame=1, candidate_index=1,
                                  previous_box=LEFT, candidate_box=LEFT, features={}), accepted=True)
        self.assertEqual(evaluate_prepared(evidence, [track], [action]).wrong, 1)
        self.assertEqual(evaluate_prepared(evidence, [track], [action], duplicate_equivalence=True).wrong, 0)


if __name__ == "__main__":
    unittest.main()
