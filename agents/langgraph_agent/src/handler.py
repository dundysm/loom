"""AgentCore Runtime entry point for the LangGraph agent.

Uses ``BedrockAgentCoreApp`` from the ``bedrock-agentcore`` SDK exactly like
the Strands and ADK handlers; only the streaming and pause/resume mechanics
underneath differ.

Wire-contract parity with Strands (``backend/app/routers/invocations.py``
consumes these shapes unchanged):

- Plain ``str``: a streamed text token.
- ``{"tool_use": {"name": str, "id": str}}``: a tool call started.
- ``{"interrupt": {"stopReason": "interrupt", "interrupts": [{"id","name","reason"}]}}``:
  the graph paused for approval. ``id`` is LangGraph's own interrupt id, so
  ``interruptResponse`` resumes that exact interrupt.
- ``{"token_info": {...}}``: OBO token metadata from MCP tool results.
- Plain string error text, same ``"\\n\\nError: ..."`` format Strands uses.

Session model: the compiled graph uses an in-memory checkpointer and every
run passes ``thread_id=<session_id>``, so conversation history and paused
approvals persist across invocations on the same container.

Known parity gaps (same as ADK): MCP elicitation (``{"elicitation": ...}``,
Methods 3/4) and per-invocation ``dynamic_mcp_servers`` are not wired; the
WebSocket handler supports the basic prompt/result/error protocol only.
"""

import logging
import os
import sys
import threading
from typing import Any, AsyncGenerator, Optional

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.types import Command

from src.agent import LoomAgent, build_agent
from src.config import load_config
from src.integrations.code_interpreter import AgentCoreCodeInterpreterTools
from src.integrations.mcp_client import drain_token_info_events, reset_token_info_state, set_user_access_token
from src.telemetry import TelemetryCallbackHandler, trace_invocation

_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
    force=True,
)

app = BedrockAgentCoreApp()
logger = app.logger

# Module-level state, initialized once at cold start
_agent: Optional[LoomAgent] = None
_telemetry = TelemetryCallbackHandler()


def _prewarm_code_interpreter(ci: AgentCoreCodeInterpreterTools) -> None:
    """Pre-warm the code interpreter session in a background thread.

    Session creation takes 30-60s; doing it at startup avoids the AgentCore
    Runtime HTTP streaming timeout on the first tool call.
    """
    try:
        logger.info("Pre-warming code interpreter session")
        ci.prewarm()
        logger.info("Code interpreter session pre-warm complete")
    except Exception as e:
        logger.warning("Code interpreter pre-warm failed: %s. Session will be created on first use.", e)


async def _get_agent() -> LoomAgent:
    """Get or initialize the singleton agent."""
    global _agent
    if _agent is None:
        _agent = await build_agent(load_config())
        if _agent.code_interpreter is not None:
            threading.Thread(target=_prewarm_code_interpreter, args=(_agent.code_interpreter,), daemon=True).start()
        logger.info("Agent initialized successfully")
    return _agent


def _run_config(session_id: str, actor_id: str, runtime_model_id: Optional[str]) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": session_id or "default",
            "actor_id": actor_id,
            "model_id": runtime_model_id,
        },
        "callbacks": [_telemetry],
    }


def _chunk_text(chunk: AIMessageChunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    texts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            texts.append(block["text"])
    return "".join(texts)


def _interrupt_event(interrupts: Any) -> dict[str, Any]:
    """Translate LangGraph ``Interrupt`` objects into Strands' interrupt shape."""
    items = []
    for intr in interrupts:
        value = intr.value if isinstance(intr.value, dict) else {"reason": str(intr.value)}
        items.append({"id": intr.id, "name": value.get("tool_name", ""), "reason": value})
    return {"interrupt": {"stopReason": "interrupt", "interrupts": items}}


async def _run_turn(
    graph_input: Any,
    session_id: str,
    actor_id: str,
    runtime_model_id: Optional[str] = None,
) -> AsyncGenerator[Any, None]:
    """Run the graph once, yielding events in the Strands handler's shapes."""
    agent = await _get_agent()
    run_config = _run_config(session_id, actor_id, runtime_model_id)
    reset_token_info_state()
    seen_tool_ids: set[str] = set()

    with trace_invocation(invocation_id=session_id or None):
        try:
            async for mode, data in agent.graph.astream(
                graph_input, run_config, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    chunk, metadata = data
                    if metadata.get("langgraph_node") != "agent" or not isinstance(chunk, AIMessageChunk):
                        continue
                    for tc in chunk.tool_call_chunks or []:
                        if tc.get("name") and tc.get("id") and tc["id"] not in seen_tool_ids:
                            seen_tool_ids.add(tc["id"])
                            logger.info("Tool call detected: %s", tc["name"])
                            yield {"tool_use": {"name": tc["name"], "id": tc["id"]}}
                    text = _chunk_text(chunk)
                    if text:
                        yield text
                    continue

                # mode == "updates"
                if "__interrupt__" in data:
                    event = _interrupt_event(data["__interrupt__"])
                    logger.info("Agent paused for approval: %d interrupt(s)", len(event["interrupt"]["interrupts"]))
                    yield event
                    continue
                if "tools" in data:
                    for token_info in drain_token_info_events():
                        yield {"token_info": token_info}
                agent_update = data.get("agent") or {}
                for msg in agent_update.get("messages", []):
                    # Non-streaming providers never emit chunks; surface
                    # their tool calls from the final message instead.
                    for tc in getattr(msg, "tool_calls", None) or []:
                        if tc.get("id") and tc["id"] not in seen_tool_ids:
                            seen_tool_ids.add(tc["id"])
                            yield {"tool_use": {"name": tc["name"], "id": tc["id"]}}
        except Exception as e:
            logger.exception("Agent stream error session_id=%s (%s): %s", session_id, type(e).__name__, e)
            detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            yield f"\n\nError: {detail}"


async def _build_resume_command(agent: LoomAgent, session_id: str, resume: dict[str, Any]) -> Command:
    """Build the ``Command`` that resumes a paused approval.

    Targets the interrupt by id when it is still pending; otherwise resumes
    the next pending interrupt with the bare value.
    """
    interrupt_id = resume.get("interruptId", "")
    response_value = resume.get("response", "n")
    state = await agent.graph.aget_state({"configurable": {"thread_id": session_id or "default"}})
    pending_ids = {i.id for i in (state.interrupts or ())}
    if interrupt_id and interrupt_id in pending_ids:
        return Command(resume={interrupt_id: response_value})
    return Command(resume=response_value)


@app.entrypoint
async def invoke(payload: dict[str, Any]) -> AsyncGenerator[Any, None]:
    """Handle an AgentCore Runtime invocation with streaming response.

    Same two modes as the Strands handler:
    1. Normal prompt: stream text back.
    2. ``interruptResponse``: resume a paused approval.
    """
    session_id = payload.get("session_id", "")
    actor_id = payload.get("actor_id") or "loom-agent"
    runtime_model_id = payload.get("model_id")
    set_user_access_token(payload.get("user_access_token"))

    agent = await _get_agent()

    interrupt_responses = payload.get("interruptResponse")
    if interrupt_responses and isinstance(interrupt_responses, list):
        if len(interrupt_responses) > 1:
            logger.warning(
                "Received %d interruptResponse entries; resuming the first, the rest will be asked again",
                len(interrupt_responses),
            )
        resume = interrupt_responses[0].get("interruptResponse", interrupt_responses[0])
        logger.info(
            "Resuming from interrupt session_id=%s id=%s response=%s",
            session_id, resume.get("interruptId", ""), resume.get("response", "n"),
        )
        command = await _build_resume_command(agent, session_id, resume)
        async for event in _run_turn(command, session_id, actor_id, runtime_model_id):
            yield event
        return

    # Policies come from the invocation payload (sent by Loom). An invocation
    # with none must not inherit the previous one's: the runtime is a warm
    # singleton shared across callers.
    invocation_policies = payload.get("approval_policies")
    if invocation_policies and isinstance(invocation_policies, list):
        agent.approval_matcher.policies = invocation_policies
        logger.info("Injected %d approval policy(ies) from payload", len(invocation_policies))
    else:
        agent.approval_matcher.policies = []

    prompt = payload.get("prompt", "")
    logger.info("Processing invocation session_id=%s actor_id=%s model=%s", session_id, actor_id, runtime_model_id or "default")
    graph_input = {"messages": [HumanMessage(content=prompt)] if prompt else []}
    async for event in _run_turn(graph_input, session_id, actor_id, runtime_model_id):
        yield event


@app.websocket
async def ws_invoke(websocket: Any, context: Any) -> None:
    """WebSocket handler.

    Supports the same prompt/result/error message protocol as the Strands
    ``ws_invoke`` (the full reply is sent once as ``result``). MCP elicitation is
    not wired (see module docstring), so no ``elicitation`` message is sent.
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type", "prompt") != "prompt":
                continue

            prompt = data.get("prompt", "")
            session_id = data.get("session_id", "")
            actor_id = data.get("actor_id") or "loom-agent"
            runtime_model_id = data.get("model_id")
            set_user_access_token(data.get("user_access_token"))

            graph_input = {"messages": [HumanMessage(content=prompt)] if prompt else []}
            result_chunks: list[str] = []
            async for event in _run_turn(graph_input, session_id, actor_id, runtime_model_id):
                if isinstance(event, str):
                    result_chunks.append(event)
                elif isinstance(event, dict) and "tool_use" in event:
                    await websocket.send_json({"type": "tool_use", "name": event["tool_use"]["name"]})
                elif isinstance(event, dict) and "interrupt" in event:
                    await websocket.send_json({
                        "type": "error",
                        "content": "This agent requires approval for a tool call; use the HTTP invoke endpoint to resume with interruptResponse.",
                    })

            await websocket.send_json({"type": "result", "content": "".join(result_chunks)})

    except Exception as e:
        logger.exception("WebSocket handler error: %s", e)
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


def main() -> None:
    """Local development entry point for running the agent interactively."""
    import asyncio

    async def _main() -> None:
        await _get_agent()
        print("Agent ready. Type your prompt (Ctrl+C to exit):")
        while True:
            try:
                prompt = input("\n> ")
                if not prompt.strip():
                    continue
                async for event in _run_turn({"messages": [HumanMessage(content=prompt)]}, "local-dev-session", "local-dev"):
                    if isinstance(event, str):
                        print(event, end="", flush=True)
                print()
            except KeyboardInterrupt:
                print("\nExiting.")
                break
            except Exception as e:
                print(f"Error: {e}")

    asyncio.run(_main())


if __name__ == "__main__":
    app.run()
