"""HITL approval policy integration for the LangGraph agent.

Ports the policy semantics of ``agents/strands_agent/src/integrations/approval.py``
(same ``LOOM_APPROVAL_POLICIES`` env var, same policy dict shape, same
matching rules) onto LangGraph's native human-in-the-loop primitive:

- Strands: a ``BeforeToolCallEvent`` hook calls ``event.interrupt()``.
- ADK: a per-tool ``require_confirmation`` predicate pauses the runner.
- LangGraph: the tools node calls ``langgraph.types.interrupt()`` before
  running a tool that matches a policy. The graph's checkpointer persists the
  paused state per ``thread_id`` (the Loom session id), and the next
  invocation resumes it with ``Command(resume=...)``.

The resume value follows the same string convention Loom's backend already
forwards for Strands ("y"/"yes"/"t"/"approved" approve, anything else
denies). "t" additionally trusts the tool for the rest of the session; the
trust set lives in graph state (``trusted_tools``) so it is checkpointed
alongside the conversation, the same way Strands keeps it in ``agent.state``.
"""

import fnmatch
import json
import logging
import os
from typing import Any

from langgraph.types import interrupt

logger = logging.getLogger(__name__)

_APPROVE_VALUES = ("y", "yes", "t", "approved")


def _load_policies() -> list[dict[str, Any]]:
    raw = os.environ.get("LOOM_APPROVAL_POLICIES", "[]")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LOOM_APPROVAL_POLICIES")
        return []


def _matches_tool(tool_name: str, rules: list[str]) -> bool:
    if not rules:
        return True
    return any(fnmatch.fnmatch(tool_name, pattern) for pattern in rules)


def _matches_agent(policy: dict[str, Any], agent_tags: dict[str, str] | None = None) -> bool:
    scope = policy.get("agent_scope", {"type": "all"})
    scope_type = scope.get("type", "all")
    if scope_type == "all":
        return True
    if scope_type == "tag_filter" and agent_tags:
        key = scope.get("tag_key", "")
        value = scope.get("tag_value", "")
        return agent_tags.get(key) == value
    return True


class ApprovalPolicyMatcher:
    """Holds the active approval policies and matches tool calls against them.

    Same matching precedence as Strands' ``ApprovalHook``. The handler
    replaces ``policies`` on every invocation from the payload's
    ``approval_policies`` field, so an operator enabling a policy in the UI
    takes effect without a redeploy.
    """

    def __init__(
        self,
        policies: list[dict[str, Any]] | None = None,
        agent_tags: dict[str, str] | None = None,
    ) -> None:
        self.policies = policies if policies is not None else _load_policies()
        self.agent_tags = agent_tags or {}

    def find_matching_policy(self, tool_name: str) -> dict[str, Any] | None:
        for policy in self.policies:
            if not policy.get("enabled", True):
                continue
            if policy.get("policy_type") not in ("loop_hook", None):
                continue
            if not _matches_agent(policy, self.agent_tags):
                continue
            if _matches_tool(tool_name, policy.get("tool_match_rules", [])):
                return policy
        return None

    def requires_approval(self, tool_name: str, trusted_tools: set[str] | list[str] | None = None) -> bool:
        """Whether a call to ``tool_name`` must pause for approval."""
        if trusted_tools and tool_name in trusted_tools:
            return False
        policy = self.find_matching_policy(tool_name)
        if policy is None:
            return False
        if policy.get("approval_mode", "require_approval") == "notify_only":
            logger.info("Approval policy '%s' is notify_only for tool '%s'", policy.get("name", ""), tool_name)
            return False
        return True


def build_interrupt_payload(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """Build the ``reason`` payload surfaced in the ``interrupt`` event.

    Same shape the Strands and ADK handlers emit, so the frontend approval
    card renders identically.
    """
    return {
        "reason": f"Authorize {tool_name}",
        "tool_name": tool_name,
        "tool_input_summary": str(tool_args)[:500],
    }


def parse_resume_value(value: Any) -> tuple[bool, bool]:
    """Translate a resume value into ``(approved, trust)``."""
    if isinstance(value, dict):
        value = value.get("response", "n")
    text = str(value or "n").lower()
    return text in _APPROVE_VALUES, text == "t"


def request_approval(tool_name: str, tool_args: dict[str, Any]) -> tuple[bool, bool]:
    """Pause the graph until the user approves or denies this tool call.

    Must be called from inside a graph node. On first execution this raises
    LangGraph's interrupt; when the graph is resumed with
    ``Command(resume=value)`` the node re-runs and this returns the parsed
    ``(approved, trust)`` decision.
    """
    value = interrupt(build_interrupt_payload(tool_name, tool_args))
    return parse_resume_value(value)
