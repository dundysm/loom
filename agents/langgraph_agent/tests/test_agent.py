"""Tests for model selection and the LangGraph agent graph."""

import os
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from src.agent import LoomAgent, build_agent, build_graph, build_model
from src.config import (
    A2AAgentConfig,
    AgentConfig,
    IntegrationsConfig,
    MCPServerConfig,
    MemoryConfig,
    MemoryResourceConfig,
)
from src.integrations.approval import ApprovalPolicyMatcher
from tests.fakes import FakeToolChatModel, tool_call_message


def _config(**kwargs) -> AgentConfig:
    base = {"system_prompt": "Be helpful.", "model_id": "us.anthropic.claude-sonnet-4-20250514"}
    base.update(kwargs)
    return AgentConfig(**base)


@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"sunny in {city}"


_POLICY = {"name": "gate", "enabled": True, "policy_type": "loop_hook", "tool_match_rules": ["get_*"]}


class TestBuildModel(unittest.TestCase):
    """Tests for build_model provider selection."""

    @patch.dict(os.environ, {
        "AWS_REGION": "us-west-2",
        "AWS_ACCESS_KEY_ID": "testing",  # nosec B105 — dummy value so boto3 skips the metadata lookup
        "AWS_SECRET_ACCESS_KEY": "testing",  # nosec B105
    })
    def test_bedrock_default(self) -> None:
        from langchain_aws import ChatBedrockConverse

        model = build_model(_config())
        self.assertIsInstance(model, ChatBedrockConverse)
        self.assertEqual(model.model_id, "us.anthropic.claude-sonnet-4-20250514")
        self.assertEqual(model.region_name, "us-west-2")
        self.assertEqual(model.max_tokens, 4096)

    @patch("src.agent.resolve_secret", return_value="sk-test")
    def test_openai(self, _mock_secret: MagicMock) -> None:
        from langchain_openai import ChatOpenAI

        model = build_model(_config(provider="openai", model_id="gpt-4o", api_key_secret_arn="arn:secret"))
        self.assertIsInstance(model, ChatOpenAI)
        self.assertEqual(model.model_name, "gpt-4o")

    @patch("src.agent.resolve_secret", return_value="sk-ant")
    def test_anthropic(self, _mock_secret: MagicMock) -> None:
        from langchain_anthropic import ChatAnthropic

        model = build_model(_config(provider="anthropic", model_id="claude-sonnet-4", api_key_secret_arn="arn:secret"))
        self.assertIsInstance(model, ChatAnthropic)

    @patch("src.agent.resolve_secret", return_value="sk-proxy")
    def test_litellm_routes_through_proxy(self, _mock_secret: MagicMock) -> None:
        from langchain_openai import ChatOpenAI

        model = build_model(_config(
            provider="litellm", model_id="gpt-4o", api_key_secret_arn="arn:secret", base_url="https://proxy.example.com",
        ))
        self.assertIsInstance(model, ChatOpenAI)
        self.assertEqual(model.openai_api_base, "https://proxy.example.com")

    def test_litellm_requires_base_url(self) -> None:
        with self.assertRaises(ValueError):
            build_model(_config(provider="litellm", api_key_secret_arn="arn:secret"))

    def test_api_key_required(self) -> None:
        with self.assertRaises(ValueError):
            build_model(_config(provider="openai"))

    def test_unsupported_provider(self) -> None:
        with self.assertRaises(ValueError):
            build_model(_config(provider="gemini"))


class TestBuildAgent(unittest.IsolatedAsyncioTestCase):
    """Tests for build_agent integration wiring."""

    def setUp(self) -> None:
        patcher = patch("src.agent.build_model", return_value=FakeToolChatModel())
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_no_integrations(self) -> None:
        agent = await build_agent(_config())
        self.assertIsInstance(agent, LoomAgent)
        self.assertEqual(agent.tools, [])
        self.assertIsNone(agent.memory)
        self.assertIsNone(agent.code_interpreter)
        self.assertEqual(agent.approval_matcher.policies, [])

    @patch("src.agent.load_tools")
    async def test_loads_static_mcp_servers_only(self, mock_load: MagicMock) -> None:
        mock_load.return_value = [get_weather]
        servers = [
            MCPServerConfig(name="static", enabled=True, endpoint_url="https://a"),
            MCPServerConfig(name="dynamic", enabled=True, endpoint_url="https://b", dynamic_only=True),
            MCPServerConfig(name="off", enabled=False, endpoint_url="https://c"),
        ]
        agent = await build_agent(_config(integrations=IntegrationsConfig(mcp_servers=servers)))
        loaded = mock_load.call_args.args[0]
        self.assertEqual([s.name for s in loaded], ["static"])
        self.assertEqual(agent.tools, [get_weather])

    @patch("src.agent.create_a2a_tools")
    async def test_loads_enabled_a2a_agents(self, mock_a2a: MagicMock) -> None:
        mock_a2a.return_value = [get_weather]
        agents = [A2AAgentConfig(name="on", enabled=True), A2AAgentConfig(name="off", enabled=False)]
        agent = await build_agent(_config(integrations=IntegrationsConfig(a2a_agents=agents)))
        self.assertEqual([a.name for a in mock_a2a.call_args.args[0]], ["on"])
        self.assertEqual(len(agent.tools), 1)

    async def test_memory_enabled_uses_first_resource(self) -> None:
        memory = MemoryConfig(enabled=True, resources=[MemoryResourceConfig(name="m", memory_id="mem-123")])
        agent = await build_agent(_config(integrations=IntegrationsConfig(memory=memory)))
        self.assertIsNotNone(agent.memory)
        self.assertEqual(agent.memory.memory_store_id, "mem-123")

    @patch("src.agent.build_model")
    async def test_resolve_model_caches_overrides(self, mock_build: MagicMock) -> None:
        default, override = MagicMock(), MagicMock()
        mock_build.side_effect = [default, override]
        agent = await build_agent(_config())
        self.assertIs(agent.resolve_model(None), default)
        self.assertIs(agent.resolve_model("other-model"), override)
        self.assertIs(agent.resolve_model("other-model"), override)
        self.assertEqual(mock_build.call_count, 2)
        self.assertEqual(mock_build.call_args.args[0].model_id, "other-model")


class TestGraph(unittest.IsolatedAsyncioTestCase):
    """Tests for the compiled graph: tool loop, approvals, memory."""

    def _graph(self, model: FakeToolChatModel, policies=None, memory=None, tools=None):
        matcher = ApprovalPolicyMatcher(policies=policies or [])
        graph = build_graph(_config(), tools if tools is not None else [get_weather], lambda _mid: model, matcher, memory)
        return graph, matcher

    async def test_plain_reply(self) -> None:
        model = FakeToolChatModel(responses=[AIMessage(content="hello there")])
        graph, _ = self._graph(model)
        cfg = {"configurable": {"thread_id": "s1"}}
        state = await graph.ainvoke({"messages": [HumanMessage(content="hi")]}, cfg)
        self.assertEqual(state["messages"][-1].content, "hello there")
        self.assertIsInstance(model.calls[0][0], SystemMessage)
        self.assertEqual(model.calls[0][0].content, "Be helpful.")

    async def test_tool_loop(self) -> None:
        model = FakeToolChatModel(responses=[
            tool_call_message("get_weather", {"city": "Paris"}),
            AIMessage(content="It is sunny."),
        ])
        graph, _ = self._graph(model)
        state = await graph.ainvoke({"messages": [HumanMessage(content="weather?")]}, {"configurable": {"thread_id": "s1"}})
        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(tool_messages[0].content, "sunny in Paris")
        self.assertEqual(state["messages"][-1].content, "It is sunny.")

    async def test_unknown_tool_returns_error_message(self) -> None:
        model = FakeToolChatModel(responses=[tool_call_message("nope", {}), AIMessage(content="ok")])
        graph, _ = self._graph(model)
        state = await graph.ainvoke({"messages": [HumanMessage(content="x")]}, {"configurable": {"thread_id": "s1"}})
        tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        self.assertIn("not available", tool_msg.content)

    async def test_conversation_persists_per_thread(self) -> None:
        model = FakeToolChatModel(responses=[AIMessage(content="one"), AIMessage(content="two")])
        graph, _ = self._graph(model)
        cfg = {"configurable": {"thread_id": "s1"}}
        await graph.ainvoke({"messages": [HumanMessage(content="a")]}, cfg)
        state = await graph.ainvoke({"messages": [HumanMessage(content="b")]}, cfg)
        self.assertEqual([m.content for m in state["messages"]], ["a", "one", "b", "two"])

    async def test_approval_interrupt_and_approve(self) -> None:
        model = FakeToolChatModel(responses=[
            tool_call_message("get_weather", {"city": "Rome"}),
            AIMessage(content="done"),
        ])
        graph, _ = self._graph(model, policies=[_POLICY])
        cfg = {"configurable": {"thread_id": "s1"}}
        state = await graph.ainvoke({"messages": [HumanMessage(content="w")]}, cfg)
        interrupts = state["__interrupt__"]
        self.assertEqual(interrupts[0].value["tool_name"], "get_weather")
        self.assertIn("Rome", interrupts[0].value["tool_input_summary"])

        state = await graph.ainvoke(Command(resume={interrupts[0].id: "y"}), cfg)
        tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(tool_msg.content, "sunny in Rome")
        self.assertEqual(state["messages"][-1].content, "done")
        self.assertNotIn("get_weather", state.get("trusted_tools", []))

    async def test_approval_deny(self) -> None:
        model = FakeToolChatModel(responses=[tool_call_message("get_weather", {"city": "Rome"}), AIMessage(content="ok")])
        graph, _ = self._graph(model, policies=[_POLICY])
        cfg = {"configurable": {"thread_id": "s1"}}
        await graph.ainvoke({"messages": [HumanMessage(content="w")]}, cfg)
        state = await graph.ainvoke(Command(resume="n"), cfg)
        tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        self.assertIn("denied", tool_msg.content)

    async def test_trust_skips_later_approvals(self) -> None:
        model = FakeToolChatModel(responses=[
            tool_call_message("get_weather", {"city": "Rome"}, "c1"),
            AIMessage(content="first"),
            tool_call_message("get_weather", {"city": "Oslo"}, "c2"),
            AIMessage(content="second"),
        ])
        graph, _ = self._graph(model, policies=[_POLICY])
        cfg = {"configurable": {"thread_id": "s1"}}
        await graph.ainvoke({"messages": [HumanMessage(content="w")]}, cfg)
        state = await graph.ainvoke(Command(resume="t"), cfg)
        self.assertEqual(state["trusted_tools"], ["get_weather"])
        state = await graph.ainvoke({"messages": [HumanMessage(content="again")]}, cfg)
        self.assertNotIn("__interrupt__", state)
        self.assertEqual(state["messages"][-1].content, "second")

    async def test_notify_only_policy_does_not_pause(self) -> None:
        model = FakeToolChatModel(responses=[tool_call_message("get_weather", {"city": "Rome"}), AIMessage(content="ok")])
        graph, _ = self._graph(model, policies=[{**_POLICY, "approval_mode": "notify_only"}])
        state = await graph.ainvoke({"messages": [HumanMessage(content="w")]}, {"configurable": {"thread_id": "s1"}})
        self.assertNotIn("__interrupt__", state)

    async def test_memory_load_and_save_new_messages_only(self) -> None:
        memory = MagicMock()

        async def _load(messages):
            return [{"content": {"text": "likes tea"}}]

        saved = {}

        async def _save(new_messages, actor_id=None, session_id=None):
            saved["messages"] = [m.content for m in new_messages]
            saved["actor_id"] = actor_id
            saved["session_id"] = session_id

        memory.load.side_effect = _load
        memory.save.side_effect = _save
        model = FakeToolChatModel(responses=[AIMessage(content="one"), AIMessage(content="two")])
        graph, _ = self._graph(model, memory=memory)
        cfg = {"configurable": {"thread_id": "s1", "actor_id": "alice"}}
        await graph.ainvoke({"messages": [HumanMessage(content="a")]}, cfg)
        state = await graph.ainvoke({"messages": [HumanMessage(content="b")]}, cfg)
        self.assertEqual(state["memory"], [{"content": {"text": "likes tea"}}])
        self.assertEqual(saved, {"messages": ["b", "two"], "actor_id": "alice", "session_id": "s1"})


if __name__ == "__main__":
    unittest.main()
