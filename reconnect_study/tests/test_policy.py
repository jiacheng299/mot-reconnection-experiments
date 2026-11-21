from types import SimpleNamespace
import unittest

from reconnect_study.policy import RecoveryPolicy


class RecoveryPolicyTest(unittest.TestCase):
    def test_cap_counts_only_accepted_actions_and_resets_per_instance(self):
        policy = RecoveryPolicy(lambda x: x["score"], .6, 2)
        proposal = lambda value: SimpleNamespace(features={"score": value})
        self.assertEqual([policy(proposal(value)) for value in (.2, .6, .1, .8, 1.0)],
                         [False, True, False, True, False])
        self.assertEqual(policy.accepted, 2)
        self.assertTrue(RecoveryPolicy(lambda x: x["score"], .6, 2)(proposal(.9)))

    def test_can_run_a_control_that_only_caps_acceptances(self):
        policy = RecoveryPolicy(None, None, 1)
        self.assertTrue(policy(SimpleNamespace(features={})))
        self.assertFalse(policy(SimpleNamespace(features={})))

    def test_nonfinite_score_leaves_the_action_budget_available(self):
        policy = RecoveryPolicy(lambda x: x["score"], .6, 1)
        self.assertFalse(policy(SimpleNamespace(features={"score": float("nan")})))
        self.assertTrue(policy(SimpleNamespace(features={"score": .8})))
