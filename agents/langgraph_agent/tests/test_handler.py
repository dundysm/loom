"""Tests for the handler entry point and its wire-contract event shapes."""

import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import src.handler as handler
from src.agent import LoomAgent, build_graph
from src.config import AgentConfig
from src.integrations.approval import ApprovalPolicyMatcher
from src.integrations.mcp_client import _TOKEN_INFO_PREFIX
from tests.fakes import FakeToolChatModel, tool_call_message


@tool
def lookup(item: str) -> str:
    """Look an item up."""
    return f"found {item}"


@tool
def lookup_with_token(item: str) -> str:
    """Look an item up and report token info."""
    return f'{_TOKEN_INFO_PREFIX}{{"token_type": "obo", "credential_provider": "idp"}}\nfound {item}'


_POLICY = {"name": "gate", "enabled": True, "policy_type": "loop_hook", "tool_match_rules": ["lookup"]}


def _make_agent(model: FakeToolChatModel, tools=None) -> LoomAgent:
    config = AgentConfig(system_prompt="Be helpful.", model_id="m")
    matcher = ApprovalPolicyMatcher(policies=[])
    tools = tools if tools is not None else [lookup]
    agent = LoomAgent(config=config, graph=None, tools=tools, approval_matcher=matcher, default_model=model)  # type: ignore[arg-type]
    agent.graph = build_graph(config, tools, lambda _mid: model, matcher)
    return agent


async def _collect(payload: dict) -> list:
    return [event async for event in handler.invoke(payload)]


class TestInvoke(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests of ``invoke`` against a scripted model."""

    def _patch_agent(self, agent: LoomAgent) -> None:
        patcher = patch.object(handler, "_get_agent", AsyncMock(return_value=agent))
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_streams_text_chunks(self) -> None:
        self._patch_agent(_make_agent(FakeToolChatModel(responses=[AIMessage(content="hello big world")])))
        events = await _collect({"prompt": "hi", "session_id": "s1"})
        self.assertEqual(events, ["hello ", "big ", "world"])

    async def test_tool_use_event_then_text(self) -> None:
        model = FakeToolChatModel(responses=[
            tool_call_message("lookup", {"item": "x"}, "call-9"),
            AIMessage(content="done"),
        ])
        self._patch_agent(_make_agent(model))
        events = await _collect({"prompt": "find x", "session_id": "s1"})
        self.assertEqual(events, [{"tool_use": {"name": "lookup", "id": "call-9"}}, "done"])

    async def test_interrupt_and_resume(self) -> None:
        model = FakeToolChatModel(responses=[
            tool_call_message("lookup", {"item": "x"}),
            AIMessage(content="approved and done"),
        ])
        agent = _make_agent(model)
        self._patch_agent(agent)

        events = await _collect({"prompt": "find x", "session_id": "s1", "approval_policies": [_POLICY]})
        interrupt = events[-1]["interrupt"]
        self.assertEqual(interrupt["stopReason"], "interrupt")
        entry = interrupt["interrupts"][0]
        self.assertEqual(entry["name"], "lookup")
        self.assertEqual(entry["reason"]["reason"], "Authorize lookup")
        self.assertTrue(entry["id"])

        events = await _collect({
            "session_id": "s1",
            "interruptResponse": [{"interruptResponse": {"interruptId": entry["id"], "response": "y"}}],
        })
        self.assertEqual(events, ["approved ", "and ", "done"])

    async def test_policies_reset_when_payload_has_none(self) -> None:
        agent = _make_agent(FakeToolChatModel(responses=[AIMessage(content="a"), AIMessage(content="b")]))
        self._patch_agent(agent)
        await _collect({"prompt": "x", "session_id": "s1", "approval_policies": [_POLICY]})
        self.assertEqual(agent.approval_matcher.policies, [_POLICY])
        await _collect({"prompt": "y", "session_id": "s1"})
        self.assertEqual(agent.approval_matcher.policies, [])

    async def test_token_info_event_and_marker_stripped(self) -> None:
        model = FakeToolChatModel(responses=[
            tool_call_message("lookup_with_token", {"item": "x"}),
            AIMessage(content="ok"),
        ])
        agent = _make_agent(model, tools=[lookup_with_token])
        self._patch_agent(agent)
        events = await _collect({"prompt": "x", "session_id": "s1"})
        token_events = [e for e in events if isinstance(e, dict) and "token_info" in e]
        self.assertEqual(token_events[0]["token_info"]["credential_provider"], "idp")
        tool_result = model.calls[1][-1]
        self.assertEqual(tool_result.content, "found x")

    async def test_model_error_becomes_error_text(self) -> None:
        self._patch_agent(_make_agent(FakeToolChatModel(responses=[])))
        events = await _collect({"prompt": "x", "session_id": "s1"})
        self.assertTrue(events[-1].startswith("\n\nError: "))

    async def test_runtime_model_override_passed_to_graph(self) -> None:
        default = FakeToolChatModel(responses=[AIMessage(content="default")])
        override = FakeToolChatModel(responses=[AIMessage(content="override")])
        config = AgentConfig(system_prompt="p", model_id="m")
        matcher = ApprovalPolicyMatcher(policies=[])
        agent = LoomAgent(config=config, graph=None, tools=[], approval_matcher=matcher, default_model=default)  # type: ignore[arg-type]
        agent.model_cache[(config.provider, "other", config.base_url)] = override
        agent.graph = build_graph(config, [], agent.resolve_model, matcher)
        self._patch_agent(agent)
        events = await _collect({"prompt": "x", "session_id": "s1", "model_id": "other"})
        self.assertEqual(events, ["override"])


class TestWebSocket(unittest.IsolatedAsyncioTestCase):
    """Tests for the WebSocket handler's prompt/result protocol."""

    async def test_prompt_result(self) -> None:
        agent = _make_agent(FakeToolChatModel(responses=[
            tool_call_message("lookup", {"item": "x"}),
            AIMessage(content="all done"),
        ]))
        sent: list[dict] = []
        ws = AsyncMock()
        ws.receive_json.side_effect = [{"type": "prompt", "prompt": "x", "session_id": "s1"}, RuntimeError("closed")]
        ws.send_json.side_effect = lambda msg: sent.append(msg)
        with patch.object(handler, "_get_agent", AsyncMock(return_value=agent)):
            await handler.ws_invoke(ws, None)
        self.assertEqual(sent[0], {"type": "tool_use", "name": "lookup"})
        self.assertEqual(sent[1], {"type": "result", "content": "all done"})
        self.assertEqual(sent[2]["type"], "error")


if __name__ == "__main__":
    unittest.main()
