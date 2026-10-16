"""Tests for A2A tool creation and the message/send client."""

import json
import unittest

import httpx

from src.config import A2AAgentConfig, AuthConfig
from src.integrations import a2a_client
from src.integrations.a2a_client import (
    A2AAgentClient,
    _sanitize_tool_name,
    create_a2a_tools,
    extract_result_text,
    fetch_agent_card,
)


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestHelpers(unittest.TestCase):
    def test_sanitize_tool_name(self) -> None:
        self.assertEqual(_sanitize_tool_name("my agent.v2"), "my_agent_v2")
        self.assertEqual(_sanitize_tool_name("1st"), "a2a_1st")

    def test_extract_message_text(self) -> None:
        self.assertEqual(extract_result_text({"kind": "message", "parts": [{"kind": "text", "text": "hi"}]}), "hi")

    def test_extract_task_artifacts_then_status(self) -> None:
        task = {"kind": "task", "artifacts": [{"parts": [{"kind": "text", "text": "a"}]}, {"parts": [{"kind": "text", "text": "b"}]}]}
        self.assertEqual(extract_result_text(task), "a\nb")
        status_only = {"kind": "task", "status": {"message": {"parts": [{"kind": "text", "text": "s"}]}}}
        self.assertEqual(extract_result_text(status_only), "s")


class TestAgentCard(unittest.IsolatedAsyncioTestCase):
    async def test_agentcore_url_overridden(self) -> None:
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"name": "x", "url": "http://localhost:8081"})

        card, rpc_url = await fetch_agent_card("https://rt.example.com/agent", _mock_client(handler))
        self.assertEqual(seen, ["https://rt.example.com/agent/.well-known/agent-card.json"])
        self.assertEqual(rpc_url, "https://rt.example.com/agent/")
        self.assertEqual(card["name"], "x")

    async def test_salesforce_card_path_and_url_kept(self) -> None:
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"url": "https://api.salesforce.com/rpc"})

        base = "https://api.salesforce.com/einstein/ai-agent/v1/agents/123"
        _card, rpc_url = await fetch_agent_card(base, _mock_client(handler))
        self.assertEqual(seen, [f"{base}/v1/card"])
        self.assertEqual(rpc_url, "https://api.salesforce.com/rpc")


class TestA2AClient(unittest.IsolatedAsyncioTestCase):
    async def test_send_keeps_context_id(self) -> None:
        bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"name": "remote", "description": "Remote helper"})
            body = json.loads(request.content)
            bodies.append(body)
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {
                "kind": "message", "contextId": "ctx-1", "parts": [{"kind": "text", "text": "pong"}],
            }})

        client = A2AAgentClient(A2AAgentConfig(name="remote", enabled=True, endpoint_url="https://r"), _mock_client(handler))
        await client.resolve()
        self.assertEqual(await client.send("ping"), "pong")
        await client.send("again")
        self.assertEqual(bodies[0]["method"], "message/send")
        self.assertNotIn("contextId", bodies[0]["params"]["message"])
        self.assertEqual(bodies[1]["params"]["message"]["contextId"], "ctx-1")

    async def test_jsonrpc_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "error": {"code": -1, "message": "nope"}})

        client = A2AAgentClient(A2AAgentConfig(name="r", enabled=True, endpoint_url="https://r"), _mock_client(handler))
        client.rpc_url = "https://r/"
        self.assertIn("nope", await client.send("x"))

    async def test_oauth2_client_uses_bearer_auth(self) -> None:
        cfg = A2AAgentConfig(name="r", enabled=True, auth=AuthConfig(type="oauth2", credential_provider_name="idp"))
        client = a2a_client._build_http_client(cfg)
        self.assertIsInstance(client.auth, a2a_client._BearerAuth)
        await client.aclose()


class TestCreateA2ATools(unittest.IsolatedAsyncioTestCase):
    async def test_skips_unreachable_and_disabled(self) -> None:
        original = a2a_client._build_http_client

        def handler(request: httpx.Request) -> httpx.Response:
            if "bad" in str(request.url):
                return httpx.Response(503)
            return httpx.Response(200, json={"name": "good", "description": "Good agent"})

        a2a_client._build_http_client = lambda _cfg: _mock_client(handler)
        try:
            tools = await create_a2a_tools([
                A2AAgentConfig(name="good-agent", enabled=True, endpoint_url="https://good"),
                A2AAgentConfig(name="bad", enabled=True, endpoint_url="https://bad"),
                A2AAgentConfig(name="off", enabled=False, endpoint_url="https://good"),
            ])
        finally:
            a2a_client._build_http_client = original
        self.assertEqual([t.name for t in tools], ["good-agent"])
        self.assertEqual(tools[0].description, "Good agent")
        self.assertEqual(list(tools[0].args), ["message"])


if __name__ == "__main__":
    unittest.main()
