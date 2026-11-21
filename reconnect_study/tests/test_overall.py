from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from reconnect_study.data_io import Sequence
from reconnect_study.overall import OverallEvaluator, evaluate_overall, write_mot_rows
from reconnect_study.tracker import TrackingResult, TrackObservation


BOX = (10.0, 20.0, 30.0, 60.0)


def subject(name, frames, gt=None, *, source_id=None, fps=20.0):
    return Sequence(name, source_id or name, frames, fps, gt, None, "validation")


def observations(frames, track_id=19):
    return tuple(TrackObservation(frame, track_id, BOX, 0.9) for frame in frames)


def evidence(frames):
    return SimpleNamespace(truth={frame: {100001: BOX} for frame in frames})


class OverallEvaluationTest(unittest.TestCase):
    def test_identity_switch_is_measured_on_tracker_output(self):
        result = TrackingResult(observations([1]) + observations([2], track_id=27), ())
        evaluator = OverallEvaluator()
        measured = evaluator.add(subject("switch", 2), evidence([1, 2]), result)

        self.assertEqual(measured["IDSW"], 1)
        self.assertAlmostEqual(measured["AssA"], 50.0)
        self.assertAlmostEqual(measured["HOTA"], 100 / 2 ** 0.5)
        self.assertAlmostEqual(measured["IDF1"], 50.0)

    def test_aggregate_uses_official_counts_instead_of_mean_sequence_scores(self):
        rows = [
            (subject("empty", 1), evidence([1]), TrackingResult((), ())),
            (subject("long", 9), evidence(range(1, 10)), TrackingResult(observations(range(1, 10)), ())),
        ]
        report = evaluate_overall(rows)

        self.assertEqual(report["sequences"]["empty"]["HOTA"], 0.0)
        self.assertEqual(report["sequences"]["long"]["HOTA"], 100.0)
        self.assertAlmostEqual(report["aggregate"]["HOTA"], 100 * 0.9 ** 0.5)
        self.assertAlmostEqual(report["aggregate"]["IDF1"], 100 * 18 / 19)
        self.assertEqual(report["aggregate"]["AssA"], 100.0)
        self.assertEqual(report["aggregate"]["IDSW"], 0)
        self.assertTrue(report["trackeval_version"])

    def test_mot_evaluation_reads_raw_gt_to_ignore_distractors(self):
        with TemporaryDirectory() as folder:
            gt = Path(folder) / "gt.txt"
            gt.write_text("1,1,11,21,20,40,1,1,1\n1,2,101,101,20,20,1,8,1\n", encoding="utf-8")
            extra = TrackObservation(1, 20, (100.0, 100.0, 120.0, 120.0), 0.9)
            result = TrackingResult(observations([1]) + (extra,), ())
            rows = [(subject("raw", 1, gt), evidence([1]), result)]
            mot = evaluate_overall(rows, protocol="mot17")
            dance = evaluate_overall(rows, protocol="dancetrack")

        self.assertEqual(mot["aggregate"]["HOTA"], 100.0)
        self.assertLess(dance["aggregate"]["HOTA"], 100.0)

    def test_mot_export_converts_xyxy_to_one_origin_xywh(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "result.txt"
            write_mot_rows(output, observations([1]))
            tokens = output.read_text(encoding="utf-8").strip().split(",")
        self.assertEqual(list(map(float, tokens)), [1, 19, 11, 21, 20, 40, .9, -1, -1, -1])

    def test_personpath_native_keyframes_ignore_crowds_and_distractors(self):
        with TemporaryDirectory() as folder:
            gt = Path(folder) / "gt.txt"
            gt.write_text(
                "1,1,11,21,20,40,1,1,1\n1,2,101,101,20,20,1,8,1\n2,3,80,80,80,80,1,13,1\n",
                encoding="utf-8",
            )
            distractor = TrackObservation(1, 20, (100.0, 100.0, 120.0, 120.0), 0.9)
            inside_crowd = TrackObservation(2, 21, (100.0, 100.0, 120.0, 120.0), 0.9)
            tracking = TrackingResult(observations([1]) + (distractor, inside_crowd), ())
            rows = [(subject("native", 2, gt, fps=5.0), evidence([1]), tracking)]
            report = evaluate_overall(rows, protocol="personpath22-native-keyframe")

        self.assertEqual(report["aggregate"]["HOTA"], 100.0)
        self.assertEqual(report["aggregate"]["IDF1"], 100.0)
        self.assertIn("native", report["protocol"])
        self.assertIn("5 Hz", report["preprocessing"])

    def test_personpath_native_keyframe_protocol_rejects_dense_frame_rate(self):
        evaluator = OverallEvaluator("personpath22-native-keyframe")
        with self.assertRaisesRegex(ValueError, "PersonPath22.+5 Hz"):
            evaluator.add(subject("dense", 2, fps=24.0), evidence([1]), TrackingResult((), ()))

    def test_grouped_idswitch_totals_only_count_within_each_video_sequence(self):
        evaluator = OverallEvaluator()
        switch = TrackingResult(observations([1]) + observations([2], track_id=27), ())
        new_clip = TrackingResult(observations([1], track_id=99), ())
        other_source = TrackingResult(observations([1]) + observations([2], track_id=23), ())
        evaluator.add(subject("clip1", 2, source_id="common"), evidence([1, 2]), switch)
        evaluator.add(subject("clip2", 1, source_id="common"), evidence([1]), new_clip)
        evaluator.add(subject("video", 2), evidence([1, 2]), other_source)

        report = evaluator.summary()
        self.assertEqual(report["aggregate"]["IDSW"], 2)
        self.assertEqual(report["idsw_by_source"], {"common": 1, "video": 1})


if __name__ == "__main__":
    unittest.main()
