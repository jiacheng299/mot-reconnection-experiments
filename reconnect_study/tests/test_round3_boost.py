import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

from reconnect_study.round3_boost import boosttrack_similarity_v1, run_bytetrack_boost
from reconnect_study.round3_boost_bot import adapt_state


UPSTREAM = (Path(__file__).resolve().parents[2] / "runs" / "reconnect_supplement" /
            "round3" / "boost" / "upstream" / "BoostTrack")


def author_assoc():
    sys.path.insert(0, str(UPSTREAM))
    try:
        settings_spec = importlib.util.spec_from_file_location("default_settings", UPSTREAM / "default_settings.py")
        settings = importlib.util.module_from_spec(settings_spec)
        settings_spec.loader.exec_module(settings)
        sys.modules["default_settings"] = settings
        spec = importlib.util.spec_from_file_location("round3_author_assoc", UPSTREAM / "tracker" / "assoc.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class BoostTrackSimilarityTests(unittest.TestCase):
    @unittest.skipUnless((UPSTREAM / "tracker" / "assoc.py").is_file(), "pinned BoostTrack checkout is unavailable")
    def test_port_matches_author_matrix_elementwise(self):
        author = author_assoc()
        detections = np.array([[0, 0, 10, 20, .9], [12, 2, 22, 18, .7]], dtype=float)
        tracks = np.array([[1, 1, 11, 21, .8], [10, 0, 20, 16, .6]], dtype=float)
        mhd = np.array([[1.5, 14.0], [8.0, 2.0]], dtype=float)
        track_confidence = tracks[:, 4]
        detection_confidence = detections[:, 4]
        expected = author.associate(
            detections, tracks, .3, mahalanobis_distance=mhd,
            track_confidence=track_confidence, detection_confidence=detection_confidence,
            lambda_iou=.5, lambda_mhd=.25, lambda_shape=.25,
        )[3]
        actual = boosttrack_similarity_v1(detections, tracks, mhd, track_confidence,
                                          detection_confidence, iou_threshold=.3)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_preserves_author_height_denominator_bug(self):
        detections = np.array([[0, 0, 10, 40]], dtype=float)
        tracks = np.array([[0, 0, 20, 20]], dtype=float)
        actual = boosttrack_similarity_v1(
            np.c_[detections, np.ones(1)], np.c_[tracks, np.ones(1)], np.zeros((1, 1)),
            np.ones(1), np.ones(1), iou_threshold=0.0,
        )
        expected = 1.5 * (1 / 3) + .25 + .25 * np.exp(-(10 / 20 + 20 / 20))
        np.testing.assert_allclose(actual, [[expected]], rtol=0, atol=1e-12)

    def test_botsort_ratio_variance_includes_size_cross_covariance(self):
        xywh = np.array([[4., 5., 10., 20.]])
        covariance = np.array([[[1., 0., 0., 0.], [0., 2., 0., 0.],
                                [0., 0., 9., 6.], [0., 0., 6., 16.]]])
        mean, variance = adapt_state(xywh, covariance)
        np.testing.assert_allclose(mean, [[4., 5., 20., .5]])
        np.testing.assert_allclose(variance, [[1., 2., 16., .0175]])

    def test_host_adapter_scores_a_real_gap_proposal(self):
        detections = {1: [[0, 0, 10, 20, .9]], 3: [[0, 0, 10, 20, .9]]}
        matrices = []
        result = run_bytetrack_boost(detections, 3, sequence="fixture", track_thresh=.5,
                                     match_thresh=.8, matrix_observer=matrices.append)
        self.assertEqual(len(result.decisions), 1)
        self.assertGreaterEqual(len(matrices), 1)
        self.assertIn("similarity", matrices[-1])
        score = result.decisions[0].proposal.features["boosttrack_v1_similarity"]
        self.assertTrue(np.isfinite(score))
        self.assertGreater(score, 0)


if __name__ == "__main__":
    unittest.main()
