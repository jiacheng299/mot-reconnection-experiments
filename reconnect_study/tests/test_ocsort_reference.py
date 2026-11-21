from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from reconnect_study.data_io import Sequence
from reconnect_study.ocsort_reference import evaluate_reference, run_ocsort


BOX = (10.0, 20.0, 30.0, 60.0, 0.95)


class OfficialOCSortReferenceTest(unittest.TestCase):
    def test_vendor_retains_an_identity_across_empty_frame(self):
        detections = {1: [BOX], 3: [BOX]}
        result = run_ocsort(detections, 3)

        self.assertEqual([(row.frame, row.track_id) for row in result.tracks], [(1, 1), (3, 1)])
        np.testing.assert_allclose([row.box for row in result.tracks], [BOX[:4], BOX[:4]], rtol=0, atol=1e-6)
        self.assertEqual(result.decisions, ())
        self.assertEqual(run_ocsort(detections, 3), result)

    def test_reference_reports_recovery_and_overall_without_commitment_risk(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            gt, det = root / "gt.txt", root / "detections.txt"
            gt.write_text("1,1,11,21,20,40,1,1,1\n2,1,11,21,20,40,1,1,1\n3,1,11,21,20,40,1,1,1\n",
                          encoding="utf-8")
            det.write_text("1,-1,11,21,20,40,.95\n3,-1,11,21,20,40,.95\n", encoding="utf-8")
            item = Sequence("sample", "source", 3, 5.0, gt, det, "development")
            result = evaluate_reference([item], protocol="personpath22-native-keyframe")

        self.assertEqual(result["recovery"]["aggregate"], {
            "gaps": 1, "correct": 1, "correct_per_gap": 1.0, "outcomes": {"correct": 1},
        })
        self.assertIn("HOTA", result["overall"]["aggregate"])
        self.assertNotIn("wrong", result["recovery"]["sequences"]["sample"])
        self.assertNotIn("accepted", result["recovery"]["aggregate"])
        self.assertFalse(result["commitment_risk_available"])


if __name__ == "__main__":
    unittest.main()
