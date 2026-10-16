"""OpenTelemetry instrumentation for the Loom LangGraph agent.

Provides application-level tracing spans for invocations, tool calls, and
model calls with the same span names/attributes as
``agents/strands_agent/src/telemetry.py`` and ``agents/adk_agent/src/telemetry.py``
(``agent.invocation``, ``tool.call`` with ``tool.name``, ``model.call``), so
trace consumers see equivalent data regardless of framework.

Tool and model spans come from a LangChain callback handler
(``TelemetryCallbackHandler``) passed in the graph's run config; LangChain
invokes it around every chat model and tool run, keyed by ``run_id``, which
is LangGraph's closest analog to Strands' hook providers and ADK's plugins.
Provider configuration (exporters, resource, etc.) is handled by the
``opentelemetry-instrument`` CLI wrapper that AgentCore Runtime uses as the
process entry point. Without the wrapper the OpenTelemetry API falls back to
noop providers automatically.
"""

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode

logger = logging.getLogger(__name__)

_TRACER_SCOPE = "loom.langgraph_agent"


def get_tracer() -> trace.Tracer:
    """Return a tracer scoped to the LangGraph agent."""
    return trace.get_tracer(_TRACER_SCOPE)


@contextmanager
def trace_invocation(
    invocation_id: Optional[str] = None,
) -> Generator[Span, None, None]:
    """Create a span that wraps a full agent invocation.

    Args:
        invocation_id: Optional identifier for the invocation.

    Yields:
        The active ``Span`` so callers can annotate it further.
    """
    tracer = get_tracer()
    attributes: dict[str, str] = {}
    if invocation_id:
        attributes["agent.invocation_id"] = invocation_id

    with tracer.start_as_current_span(
        "agent.invocation", attributes=attributes
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


class TelemetryCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that creates OTEL spans for tool and model calls.

    Spans are keyed by LangChain's ``run_id`` so concurrent tool calls
    (LangGraph can run several tool calls from one model turn) each close
    their own span. Errors set ERROR status and record the exception, the
    same shape Strands and ADK produce.
    """

    def __init__(self) -> None:
        super().__init__()
        self._spans: dict[UUID, Span] = {}

    def _end(self, run_id: UUID, error: Optional[BaseException] = None) -> None:
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        if error is not None:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
        span.end()

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        self._spans[run_id] = get_tracer().start_span("tool.call", attributes={"tool.name": name})

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._spans[run_id] = get_tracer().start_span("model.call")

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._spans[run_id] = get_tracer().start_span("model.call")

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error)
