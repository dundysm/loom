"""Tests for OpenTelemetry span structure."""

import unittest
import uuid

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from src.telemetry import TelemetryCallbackHandler, trace_invocation

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


class TestTelemetry(unittest.TestCase):
    def setUp(self) -> None:
        _exporter.clear()

    def test_invocation_span(self) -> None:
        with trace_invocation("inv-1"):
            pass
        span = _exporter.get_finished_spans()[0]
        self.assertEqual(span.name, "agent.invocation")
        self.assertEqual(span.attributes["agent.invocation_id"], "inv-1")

    def test_invocation_span_records_error(self) -> None:
        with self.assertRaises(ValueError):
            with trace_invocation():
                raise ValueError("boom")
        span = _exporter.get_finished_spans()[0]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

    def test_tool_span(self) -> None:
        handler = TelemetryCallbackHandler()
        run_id = uuid.uuid4()
        handler.on_tool_start({"name": "lookup"}, "{}", run_id=run_id)
        handler.on_tool_end("ok", run_id=run_id)
        span = _exporter.get_finished_spans()[0]
        self.assertEqual(span.name, "tool.call")
        self.assertEqual(span.attributes["tool.name"], "lookup")

    def test_tool_error_span(self) -> None:
        handler = TelemetryCallbackHandler()
        run_id = uuid.uuid4()
        handler.on_tool_start({"name": "lookup"}, "{}", run_id=run_id)
        handler.on_tool_error(RuntimeError("bad"), run_id=run_id)
        span = _exporter.get_finished_spans()[0]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

    def test_concurrent_tool_spans_close_independently(self) -> None:
        handler = TelemetryCallbackHandler()
        a, b = uuid.uuid4(), uuid.uuid4()
        handler.on_tool_start({"name": "a"}, "", run_id=a)
        handler.on_tool_start({"name": "b"}, "", run_id=b)
        handler.on_tool_end("", run_id=b)
        self.assertEqual([s.attributes["tool.name"] for s in _exporter.get_finished_spans()], ["b"])
        handler.on_tool_end("", run_id=a)
        self.assertEqual(len(_exporter.get_finished_spans()), 2)

    def test_model_span(self) -> None:
        handler = TelemetryCallbackHandler()
        run_id = uuid.uuid4()
        handler.on_chat_model_start({}, [[]], run_id=run_id)
        handler.on_llm_end(None, run_id=run_id)
        self.assertEqual(_exporter.get_finished_spans()[0].name, "model.call")

    def test_model_error_span(self) -> None:
        handler = TelemetryCallbackHandler()
        run_id = uuid.uuid4()
        handler.on_chat_model_start({}, [[]], run_id=run_id)
        handler.on_llm_error(RuntimeError("throttled"), run_id=run_id)
        self.assertEqual(_exporter.get_finished_spans()[0].status.status_code, StatusCode.ERROR)

    def test_unknown_run_id_is_ignored(self) -> None:
        TelemetryCallbackHandler().on_tool_end("", run_id=uuid.uuid4())
        self.assertEqual(_exporter.get_finished_spans(), ())


if __name__ == "__main__":
    unittest.main()
