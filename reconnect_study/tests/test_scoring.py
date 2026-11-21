import unittest

from reconnect_study.scoring import BASE_FEATURES, design, fit


def features(iou, margin):
    sample = dict.fromkeys(BASE_FEATURES, 0.25)
    sample.update(iou=iou, row_margin=margin, column_margin=margin, gap_frames=2)
    return sample


class ReconnectScoringTest(unittest.TestCase):
    def test_structured_basis_responds_to_disagreement(self):
        agreeing = features(.8, .5)
        disagreeing = features(.8, -.5)
        self.assertEqual(len(design(agreeing, False)), len(BASE_FEATURES))
        self.assertGreater(len(design(agreeing, True)), len(BASE_FEATURES))
        self.assertNotEqual(design(agreeing, True), design(disagreeing, True))

    def test_train_model_separates_simple_source_paired_examples(self):
        good, bad = features(.9, .7), features(.2, -.4)
        model, record = fit([(source, sample, label) for source in ("a", "b")
                             for sample, label in ((good, True), (bad, False))],
                            structured=True, device="cpu", iterations=50)
        self.assertGreater(model(good), model(bad))
        self.assertEqual(record["source_count"], 2)
        self.assertEqual(record["device"], "cpu")
