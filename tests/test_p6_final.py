import unittest

from scripts.p6_data import archive_member, FINAL_SEQUENCES
from scripts.p6_cases import failure_cause


class P6DataBoundaryTest(unittest.TestCase):
    def test_only_the_frozen_mot17_holdout_is_accepted(self):
        self.assertEqual(archive_member("MOT17/train/MOT17-11-FRCNN/img1/000001.jpg", "MOT17",
                                        FINAL_SEQUENCES["MOT17"]),
                         ("MOT17-11-FRCNN", "img1", "000001.jpg"))
        self.assertIsNone(archive_member("MOT17/train/MOT17-10-FRCNN/gt/gt.txt", "MOT17",
                                         FINAL_SEQUENCES["MOT17"]))

    def test_dance_submission_images_never_extract_any_ground_truth(self):
        self.assertEqual(archive_member("test1/dancetrack0003/seqinfo.ini", "DanceTrack",
                                        ("dancetrack0003",)), ("dancetrack0003", "seqinfo.ini"))
        self.assertIsNone(archive_member("test1/dancetrack0003/gt/gt.txt", "DanceTrack", ("dancetrack0003",)))
        self.assertIsNone(archive_member("test1/dancetrack0003/img1/../gt.txt", "DanceTrack", ("dancetrack0003",)))

    def test_failure_cause_is_first_missing_step(self):
        self.assertEqual(failure_cause(False, set(), set(), set(), None), 'detector')
        self.assertEqual(failure_cause(True, {3}, {4}, {4}, 4), 'gate')
        self.assertEqual(failure_cause(True, {3}, {3, 4}, {4}, 4), 'search')
        self.assertEqual(failure_cause(True, {3}, {3, 4}, {3, 4}, 4), 'value')
        self.assertEqual(failure_cause(True, {3}, {3, 4}, {3, 4}, 3), 'output-match')
