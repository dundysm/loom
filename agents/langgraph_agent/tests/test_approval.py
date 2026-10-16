"""Tests for approval policy matching and resume-value parsing."""

import json
import os
import unittest
from unittest.mock import patch

from src.integrations.approval import (
    ApprovalPolicyMatcher,
    build_interrupt_payload,
    parse_resume_value,
)


class TestApprovalPolicyMatcher(unittest.TestCase):
    """Policy semantics mirror Strands' ApprovalHook."""

    def test_loads_policies_from_env(self) -> None:
        policies = [{"name": "p", "tool_match_rules": ["x"]}]
        with patch.dict(os.environ, {"LOOM_APPROVAL_POLICIES": json.dumps(policies)}):
            self.assertEqual(ApprovalPolicyMatcher().policies, policies)

    def test_invalid_env_json_yields_no_policies(self) -> None:
        with patch.dict(os.environ, {"LOOM_APPROVAL_POLICIES": "{not json"}):
            self.assertEqual(ApprovalPolicyMatcher().policies, [])

    def test_glob_match(self) -> None:
        matcher = ApprovalPolicyMatcher(policies=[{"tool_match_rules": ["delete_*"]}])
        self.assertIsNotNone(matcher.find_matching_policy("delete_file"))
        self.assertIsNone(matcher.find_matching_policy("read_file"))

    def test_empty_rules_match_all(self) -> None:
        matcher = ApprovalPolicyMatcher(policies=[{"tool_match_rules": []}])
        self.assertIsNotNone(matcher.find_matching_policy("anything"))

    def test_disabled_and_other_policy_types_ignored(self) -> None:
        matcher = ApprovalPolicyMatcher(policies=[
            {"enabled": False, "tool_match_rules": []},
            {"policy_type": "tool_context", "tool_match_rules": []},
        ])
        self.assertIsNone(matcher.find_matching_policy("x"))

    def test_tag_filter_scope(self) -> None:
        policy = {"agent_scope": {"type": "tag_filter", "tag_key": "env", "tag_value": "prod"}, "tool_match_rules": []}
        self.assertIsNotNone(ApprovalPolicyMatcher(policies=[policy], agent_tags={"env": "prod"}).find_matching_policy("x"))
        self.assertIsNone(ApprovalPolicyMatcher(policies=[policy], agent_tags={"env": "dev"}).find_matching_policy("x"))

    def test_requires_approval(self) -> None:
        matcher = ApprovalPolicyMatcher(policies=[{"tool_match_rules": ["x"]}])
        self.assertTrue(matcher.requires_approval("x"))
        self.assertFalse(matcher.requires_approval("x", trusted_tools={"x"}))
        self.assertFalse(matcher.requires_approval("y"))

    def test_notify_only_does_not_require_approval(self) -> None:
        matcher = ApprovalPolicyMatcher(policies=[{"tool_match_rules": [], "approval_mode": "notify_only"}])
        self.assertFalse(matcher.requires_approval("x"))


class TestResumeValues(unittest.TestCase):
    """Resume strings follow the convention Loom forwards for Strands."""

    def test_approve_values(self) -> None:
        for value in ("y", "yes", "approved", "Y"):
            self.assertEqual(parse_resume_value(value), (True, False))

    def test_trust(self) -> None:
        self.assertEqual(parse_resume_value("t"), (True, True))

    def test_deny(self) -> None:
        for value in ("n", "no", "", None):
            self.assertEqual(parse_resume_value(value), (False, False))

    def test_dict_value(self) -> None:
        self.assertEqual(parse_resume_value({"response": "y"}), (True, False))

    def test_interrupt_payload_shape(self) -> None:
        payload = build_interrupt_payload("rm", {"path": "/" * 600})
        self.assertEqual(payload["reason"], "Authorize rm")
        self.assertEqual(payload["tool_name"], "rm")
        self.assertEqual(len(payload["tool_input_summary"]), 500)


if __name__ == "__main__":
    unittest.main()
