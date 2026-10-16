"""LangGraph agent initialization and configuration.

Ports ``_build_model``/``build_agent`` from the Strands and ADK agents onto
a LangGraph ``StateGraph``:

    START -> load_memory -> agent <-> tools
                              |
                              v
                         save_memory -> END

- ``agent`` calls the chat model (with tools bound) on the conversation.
- ``tools`` runs the tool calls from the last model turn. Before a tool
  that matches an approval policy runs, it pauses the graph with LangGraph's
  ``interrupt()`` (see ``integrations/approval.py``).
- ``load_memory``/``save_memory`` wrap each invocation with AgentCore
  Memory retrieval and event creation (see ``integrations/memory.py``).

The graph is compiled with an in-memory checkpointer and run with
``thread_id`` set to the Loom session id, so the conversation and any paused
approval survive across invocations on the same long-running container,
mirroring the Strands module-level agent singleton and ADK's
``InMemorySessionService``.
"""

import asyncio
import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from src.config import AgentConfig
from src.integrations.a2a_client import create_a2a_tools
from src.integrations.approval import ApprovalPolicyMatcher, request_approval
from src.integrations.code_interpreter import AgentCoreCodeInterpreterTools, build_code_interpreter_tools
from src.integrations.mcp_client import extract_token_info_from_tool_result, load_tools
from src.integrations.memory import MemoryManager
from src.integrations.secrets import resolve_secret

logger = logging.getLogger(__name__)


def _merge_trusted(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer for ``trusted_tools``: union, order preserved."""
    merged = list(left or [])
    for name in right or []:
        if name not in merged:
            merged.append(name)
    return merged


class AgentState(TypedDict, total=False):
    """Graph state, checkpointed per ``thread_id`` (Loom session id)."""

    messages: Annotated[list[AnyMessage], add_messages]
    memory: list[dict[str, Any]]
    turn_start: int
    trusted_tools: Annotated[list[str], _merge_trusted]


def build_model(config: AgentConfig) -> BaseChatModel:
    """Instantiate the LangChain chat model selected by ``config.provider``.

    Supports ``bedrock`` (default, IAM-authenticated through boto3, no API
    key needed, same as Strands' ``BedrockModel``), ``openai``,
    ``anthropic``, and ``litellm``. The latter three require an API key,
    resolved once from Secrets Manager via ``config.api_key_secret_arn``.
    A LiteLLM proxy exposes an OpenAI-compatible API, so ``litellm`` uses
    ``ChatOpenAI`` pointed at ``config.base_url``; this always routes
    through the proxy rather than letting a client library pick an upstream
    provider from the model id.
    """
    provider = config.provider or "bedrock"

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        region = os.environ.get("AWS_REGION", "us-east-1")
        return ChatBedrockConverse(model=config.model_id, region_name=region, max_tokens=config.max_tokens)

    if provider not in ("openai", "anthropic", "litellm"):
        raise ValueError(f"Unsupported model provider: '{provider}'")

    if not config.api_key_secret_arn:
        raise ValueError(f"Provider '{provider}' requires 'api_key_secret_arn' to be configured")
    if provider == "litellm" and not config.base_url:
        raise ValueError("Provider 'litellm' requires 'base_url' (the LiteLLM proxy URL) to be configured")
    api_key = resolve_secret(config.api_key_secret_arn)

    # Without an explicit timeout, a network path that accepts the TCP
    # connection but never responds hangs indefinitely instead of raising.
    # Same rationale and env var as Strands and ADK.
    request_timeout = float(os.environ.get("LOOM_MODEL_REQUEST_TIMEOUT_SECONDS", "30"))

    kwargs: dict[str, Any] = {
        "model": config.model_id,
        "api_key": api_key,
        "timeout": request_timeout,
        "max_tokens": config.max_tokens,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**kwargs)

    from langchain_openai import ChatOpenAI

    if provider == "litellm":
        logger.info(
            "LiteLLM client configured: base_url=%s timeout=%s api_key_len=%d",
            config.base_url, request_timeout, len(api_key or ""),
        )
    return ChatOpenAI(**kwargs)


@dataclass
class LoomAgent:
    """Everything the handler needs to run the agent."""

    config: AgentConfig
    graph: CompiledStateGraph
    tools: list[BaseTool]
    approval_matcher: ApprovalPolicyMatcher
    default_model: BaseChatModel
    code_interpreter: Optional[AgentCoreCodeInterpreterTools] = None
    memory: Optional[MemoryManager] = None
    model_cache: dict[tuple[str, str, str], BaseChatModel] = field(default_factory=dict)

    def resolve_model(self, runtime_model_id: Optional[str] = None) -> BaseChatModel:
        """Return the default model, or a cached model for a runtime override."""
        if not runtime_model_id or runtime_model_id == self.config.model_id:
            return self.default_model
        cache_key = (self.config.provider, runtime_model_id, self.config.base_url)
        if cache_key not in self.model_cache:
            self.model_cache[cache_key] = build_model(dataclasses.replace(self.config, model_id=runtime_model_id))
            logger.info(
                "Created cached model for runtime override: provider=%s model_id=%s",
                self.config.provider, runtime_model_id,
            )
        return self.model_cache[cache_key]


def _last_human_index(messages: list[AnyMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return len(messages)


def build_graph(
    agent_config: AgentConfig,
    tools: list[BaseTool],
    resolve_model: Any,
    approval_matcher: ApprovalPolicyMatcher,
    memory: Optional[MemoryManager] = None,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    """Compile the agent graph.

    ``resolve_model`` is called with the run's ``configurable.model_id``
    (possibly None) and returns the chat model to use, so a per-invocation
    model override needs no graph rebuild.
    """
    tools_by_name = {t.name: t for t in tools}

    async def load_memory(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        update: dict[str, Any] = {"turn_start": _last_human_index(messages)}
        if memory is not None:
            records = await memory.load(messages)
            update["memory"] = records
        return update

    async def agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        model = resolve_model((config.get("configurable") or {}).get("model_id"))
        runnable = model.bind_tools(tools) if tools else model
        messages = [SystemMessage(content=agent_config.system_prompt), *state.get("messages", [])]
        response = await runnable.ainvoke(messages, config)
        return {"messages": [response]}

    async def _run_tool(call: dict[str, Any], config: RunnableConfig) -> ToolMessage:
        tool = tools_by_name.get(call["name"])
        if tool is None:
            return ToolMessage(content=f"Error: tool '{call['name']}' is not available.", tool_call_id=call["id"], name=call["name"], status="error")
        try:
            result = await tool.ainvoke({**call, "type": "tool_call"}, config)
        except Exception as e:
            logger.warning("Tool '%s' failed: %s", call["name"], e)
            return ToolMessage(content=f"Error: {type(e).__name__}: {e}", tool_call_id=call["id"], name=call["name"], status="error")
        if not isinstance(result, ToolMessage):
            result = ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"])
        result.content = extract_token_info_from_tool_result(result.content)
        return result

    async def run_tools(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", []) or [])
        trusted = set(state.get("trusted_tools") or [])
        newly_trusted: list[str] = []
        approved_calls: list[dict[str, Any]] = []
        denied: list[ToolMessage] = []

        # Approvals are requested in call order. On resume LangGraph re-runs
        # this node and replays earlier interrupt answers by position, so
        # each interrupt() returns the answer for its own call.
        for call in calls:
            name = call["name"]
            if approval_matcher.requires_approval(name, trusted):
                approved, trust = request_approval(name, call.get("args", {}))
                if not approved:
                    logger.info("User denied tool call '%s'", name)
                    denied.append(ToolMessage(
                        content=f"User denied permission to run {name}.",
                        tool_call_id=call["id"], name=name, status="error",
                    ))
                    continue
                if trust:
                    trusted.add(name)
                    newly_trusted.append(name)
            approved_calls.append(call)

        results = await asyncio.gather(*(_run_tool(c, config) for c in approved_calls))
        by_id = {m.tool_call_id: m for m in [*results, *denied]}
        ordered = [by_id[c["id"]] for c in calls if c["id"] in by_id]
        update: dict[str, Any] = {"messages": ordered}
        if newly_trusted:
            update["trusted_tools"] = newly_trusted
        return update

    async def save_memory(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        if memory is None:
            return {}
        messages = state.get("messages", [])
        start = state.get("turn_start", len(messages))
        configurable = config.get("configurable") or {}
        await memory.save(
            messages[start:],
            actor_id=configurable.get("actor_id"),
            session_id=configurable.get("thread_id"),
        )
        return {}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "save_memory"

    builder = StateGraph(AgentState)
    builder.add_node("load_memory", load_memory)
    builder.add_node("agent", agent)
    builder.add_node("tools", run_tools)
    builder.add_node("save_memory", save_memory)
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "save_memory": "save_memory"})
    builder.add_edge("tools", "agent")
    builder.add_edge("save_memory", END)
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


async def build_agent(config: AgentConfig) -> LoomAgent:
    """Build the LangGraph agent from the provided configuration.

    Async because MCP tool listing and A2A agent-card resolution need
    network I/O. The handler calls this on the first invocation, so the
    AgentCore request context (workload access token) is available for
    authenticated MCP servers, the same reason Strands defers MCP init.
    """
    model = build_model(config)
    logger.info("Initialized model for provider=%s model_id=%s", config.provider, config.model_id)

    tools: list[BaseTool] = []
    ci_tools: Optional[AgentCoreCodeInterpreterTools] = None

    static_mcp_servers = [s for s in config.integrations.mcp_servers if s.enabled and not s.dynamic_only]
    if static_mcp_servers:
        tools.extend(await load_tools(static_mcp_servers))

    enabled_a2a_agents = [a for a in config.integrations.a2a_agents if a.enabled]
    if enabled_a2a_agents:
        a2a_tools = await create_a2a_tools(enabled_a2a_agents)
        tools.extend(a2a_tools)
        logger.info("Loaded %d A2A tool(s)", len(a2a_tools))

    if config.integrations.code_interpreter.enabled:
        ci_function_tools, ci_tools = build_code_interpreter_tools(config.integrations.code_interpreter)
        tools.extend(ci_function_tools)
        logger.info(
            "Loaded %d Code Interpreter tool(s) (region=%s)",
            len(ci_function_tools), config.integrations.code_interpreter.region or "default",
        )

    # Always attached; a no-op until the handler injects policies from the
    # invocation payload (same reasoning as the ADK agent).
    approval_matcher = ApprovalPolicyMatcher()

    memory: Optional[MemoryManager] = None
    if config.integrations.memory.enabled:
        memory_store_id = None
        if config.integrations.memory.resources:
            memory_store_id = config.integrations.memory.resources[0].memory_id
        memory = MemoryManager(memory_store_id=memory_store_id)
        logger.info("Enabled AgentCore Memory (store_id=%s)", memory_store_id)

    loom_agent = LoomAgent(
        config=config,
        graph=None,  # type: ignore[arg-type]  # set below; needs resolve_model bound to this instance
        tools=tools,
        approval_matcher=approval_matcher,
        default_model=model,
        code_interpreter=ci_tools,
        memory=memory,
    )
    loom_agent.graph = build_graph(config, tools, loom_agent.resolve_model, approval_matcher, memory)
    logger.info("Agent initialized with %d tool(s)", len(tools))
    return loom_agent
