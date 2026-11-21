import math
import unittest

from reconnect_study.risk import VideoCounts, certify_policies


def videos(count, wrong=0, accepted=1):
    return [VideoCounts(str(index), wrong, accepted) for index in range(count)]


class CertificationTests(unittest.TestCase):
    def certify(self, rows, **kwargs):
        return certify_policies({"policy": rows}, alpha=0.05, delta=0.05, **kwargs)["policy"]

    def test_zero_error_boundary_requires_independent_source_videos(self):
        for method in ("event_weighted", "active_video"):
            with self.subTest(method=method):
                weak = self.certify(videos(58), method=method, acceptance_cap=1)
                strong = self.certify(videos(59), method=method, acceptance_cap=1)
                self.assertFalse(weak.certified)
                self.assertTrue(strong.certified)
                self.assertAlmostEqual(strong.p_value, 0.95**59)
                self.assertAlmostEqual(strong.risk_upper_bound, 1 - 0.05 ** (1 / 59))

    def test_event_weighted_risk_uses_all_actions_and_a_valid_cap(self):
        rows = [VideoCounts("large", 10, 100), VideoCounts("small", 1, 1), VideoCounts("empty", 0, 0)]
        weighted = self.certify(rows, acceptance_cap=100)
        active = self.certify(rows, method="active_video")
        self.assertAlmostEqual(weighted.empirical_risk, 11 / 101)
        self.assertAlmostEqual(active.empirical_risk, 0.55)
        self.assertEqual(weighted.source_count, 3)
        self.assertEqual(weighted.active_source_count, 2)
        self.assertEqual(weighted.accepted, 101)
        with self.assertRaisesRegex(ValueError, "cap"):
            self.certify(rows, acceptance_cap=99)

    def test_sparse_acceptance_cannot_create_an_informative_event_certificate(self):
        sparse = self.certify(videos(59), acceptance_cap=1000)
        self.assertFalse(sparse.certified)
        self.assertGreater(sparse.risk_upper_bound, 0.9)

    def test_betting_mixture_improves_power_when_many_sources_have_no_errors(self):
        rows = [VideoCounts(str(i), 0, 2 if i < 80 else 0) for i in range(100)]
        settings = dict(alpha=0.10, delta=0.05, acceptance_cap=4)
        betting = certify_policies({"p": rows}, **settings, test="betting")["p"]
        kl = certify_policies({"p": rows}, **settings)["p"]
        expected_e = 0.5 * (0.925 / 0.9) ** 80 + 0.5 * (0.95 / 0.9) ** 80
        self.assertEqual(betting.test, "betting")
        self.assertAlmostEqual(betting.p_value, 1 / expected_e)
        self.assertTrue(betting.certified)
        self.assertLessEqual(betting.risk_upper_bound, 0.1)
        self.assertFalse(kl.certified)

    def test_betting_active_uses_full_fractional_loss_and_ignores_inactive_sources(self):
        rows = videos(40, wrong=1, accepted=4) + [VideoCounts("inactive", 0, 0)]
        result = certify_policies({"p": rows}, alpha=0.5, delta=0.05,
                                  test="betting", method="active_video")["p"]
        self.assertAlmostEqual(result.p_value, 1 / (0.5 * (1.25**40 + 1.5**40)))
        self.assertEqual(result.source_count, 41)
        self.assertEqual(result.active_source_count, 40)
        self.assertTrue(result.certified)

    def test_betting_mixture_remains_valid_when_the_full_stake_component_is_zero(self):
        rows = videos(100) + [VideoCounts("wrong", 1, 1)]
        result = certify_policies({"p": rows}, alpha=0.1, delta=0.05,
                                  acceptance_cap=1, test="betting")["p"]
        self.assertAlmostEqual(result.p_value, 1 / (0.25 * (0.95 / 0.9) ** 100))
        all_inactive = self.certify(videos(100, accepted=0), acceptance_cap=1, test="betting")
        self.assertFalse(all_inactive.certified)
        self.assertEqual(all_inactive.p_value, 1.0)

    def test_betting_rejects_a_boundary_null_with_probability_at_most_delta(self):
        probabilities = []
        for errors in range(41):
            rows = [VideoCounts(str(i), int(i < errors), 1) for i in range(40)]
            result = certify_policies({"p": rows}, alpha=0.1, delta=0.05, acceptance_cap=1,
                                      test="betting")["p"]
            if result.certified:
                probabilities.append(math.comb(40, errors) * 0.1**errors * 0.9 ** (40 - errors))
        self.assertGreater(sum(probabilities), 0)
        self.assertLessEqual(sum(probabilities), 0.05)

    def test_no_actions_cannot_certify_either_estimand(self):
        for method in ("event_weighted", "active_video", "active_video_exact"):
            with self.subTest(method=method):
                result = self.certify(videos(100, accepted=0), method=method, acceptance_cap=1)
                self.assertFalse(result.certified)
                self.assertIsNone(result.empirical_risk)
                self.assertEqual(result.risk_upper_bound, 1.0)
                self.assertEqual(result.p_value, 1.0)

    def test_bonferroni_applies_even_with_perfectly_dependent_policies(self):
        rows = videos(59)
        results = certify_policies(
            {"x": rows, "y": rows[::-1]}, alpha=0.05, delta=0.05, acceptance_cap=1
        )
        self.assertFalse(results["x"].certified)
        self.assertAlmostEqual(results["x"].adjusted_p_value, min(1.0, 2 * 0.95**59))
        self.assertAlmostEqual(results["x"].risk_upper_bound, 1 - 0.025 ** (1 / 59))
        self.assertEqual(results["x"], results["y"])

    def test_uses_external_uniform_label_for_exact_active_video_test(self):
        rows = [VideoCounts(str(i), wrong=int(i < 2), accepted=5, sampled_wrong=i == 0) for i in range(80)]
        result = self.certify(rows, method="active_video_exact")
        expected = 0.95**80 + 80 * 0.05 * 0.95**79
        self.assertAlmostEqual(result.empirical_risk, 0.005)
        self.assertEqual(result.sampled_wrong, 1)
        self.assertAlmostEqual(result.p_value, expected)
        self.assertEqual(result.certified, expected <= 0.05)
        with self.assertRaisesRegex(ValueError, "sampled_wrong"):
            self.certify(videos(9), method="active_video_exact")
        with self.assertRaisesRegex(ValueError, "sampled_wrong"):
            self.certify([VideoCounts("x", 0, 2, sampled_wrong=True)], method="active_video_exact")

    def test_rejects_repeated_sources_and_mismatched_policy_cohorts(self):
        rows = videos(2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.certify([rows[0], rows[0]], acceptance_cap=1)
        with self.assertRaisesRegex(ValueError, "same.*source"):
            certify_policies({"a": rows, "b": rows[:1]}, alpha=0.05, delta=0.05, acceptance_cap=1)

    def test_rejects_invalid_counts_and_risk_arguments(self):
        for row in (VideoCounts("x", 2, 1), VideoCounts("x", -1, 1), VideoCounts("x", 0, 1.5)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.certify([row], acceptance_cap=2)
        with self.assertRaisesRegex(ValueError, "alpha"):
            certify_policies({"p": videos(1)}, alpha=math.nan, delta=0.05, acceptance_cap=1)
        with self.assertRaisesRegex(ValueError, "cap"):
            self.certify(videos(1))
