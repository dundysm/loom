"""Tests for MCP connection building, auth and token_info extraction."""

import unittest
from unittest.mock import MagicMock, patch

import httpx

from src.config import AuthConfig, MCPServerConfig
from src.integrations.mcp_client import (
    _TOKEN_INFO_PREFIX,
    _ApiKeyAuth,
    _OAuth2Auth,
    build_connection,
    drain_token_info_events,
    extract_token_info_from_tool_result,
    has_deferred_auth_servers,
    load_tools,
)


def _apply(auth: httpx.Auth) -> httpx.Request:
    request = httpx.Request("POST", "https://mcp.example.com")
    flow = auth.auth_flow(request)
    return next(flow)


class TestBuildConnection(unittest.TestCase):
    def test_unauthenticated(self) -> None:
        conn = build_connection(MCPServerConfig(name="s", enabled=True, endpoint_url="https://mcp"))
        self.assertEqual(conn, {"transport": "streamable_http", "url": "https://mcp"})

    def test_unsupported_transport(self) -> None:
        self.assertIsNone(build_connection(MCPServerConfig(name="s", enabled=True, transport="stdio")))

    def test_oauth2_auth(self) -> None:
        server = MCPServerConfig(
            name="s", enabled=True, endpoint_url="https://mcp",
            auth=AuthConfig(type="oauth2", credential_provider_name="idp", scopes="a b", delegation_mode="obo"),
        )
        conn = build_connection(server)
        self.assertIsInstance(conn["auth"], _OAuth2Auth)
        self.assertEqual(conn["auth"]._fetcher._oauth2_flow, "ON_BEHALF_OF_TOKEN_EXCHANGE")
        self.assertEqual(conn["auth"]._fetcher._scopes, ["a", "b"])

    @patch.object(_ApiKeyAuth, "_resolve_key", return_value="k-123")
    def test_api_key_auth(self, _mock: MagicMock) -> None:
        server = MCPServerConfig(
            name="s", enabled=True, endpoint_url="https://mcp",
            auth=AuthConfig(type="api_key", credentials_secret_arn="arn:secret", api_key_header_name="x-key"),
        )
        conn = build_connection(server)
        self.assertEqual(_apply(conn["auth"]).headers["x-key"], "k-123")

    @patch.object(_ApiKeyAuth, "_resolve_key", return_value="k-123")
    def test_api_key_authorization_header_uses_bearer(self, _mock: MagicMock) -> None:
        auth = _ApiKeyAuth("arn:secret", header_name="Authorization")
        self.assertEqual(_apply(auth).headers["Authorization"], "Bearer k-123")

    @patch.object(_ApiKeyAuth, "_resolve_key", side_effect=Exception("denied"))
    def test_api_key_failure_falls_back_to_unauthenticated(self, _mock: MagicMock) -> None:
        server = MCPServerConfig(
            name="s", enabled=True, endpoint_url="https://mcp",
            auth=AuthConfig(type="api_key", credentials_secret_arn="arn:secret"),
        )
        self.assertNotIn("auth", build_connection(server))

    def test_oauth2_auth_injects_bearer(self) -> None:
        fetcher = MagicMock()
        fetcher.fetch_bearer_token.return_value = "tok"
        self.assertEqual(_apply(_OAuth2Auth(fetcher)).headers["Authorization"], "Bearer tok")
        fetcher.fetch_bearer_token.return_value = None
        self.assertNotIn("Authorization", _apply(_OAuth2Auth(fetcher)).headers)

    def test_has_deferred_auth_servers(self) -> None:
        self.assertTrue(has_deferred_auth_servers([MCPServerConfig(name="s", enabled=True, auth=AuthConfig(type="oauth2"))]))
        self.assertFalse(has_deferred_auth_servers([MCPServerConfig(name="s", enabled=True)]))


class TestLoadTools(unittest.IsolatedAsyncioTestCase):
    @patch("src.integrations.mcp_client.load_mcp_tools")
    async def test_skips_failing_and_disabled_servers(self, mock_load: MagicMock) -> None:
        good_tool = MagicMock()

        async def _load(session, connection, server_name):
            if server_name == "bad":
                raise ConnectionError("unreachable")
            return [good_tool]

        mock_load.side_effect = _load
        tools = await load_tools([
            MCPServerConfig(name="good", enabled=True, endpoint_url="https://a"),
            MCPServerConfig(name="bad", enabled=True, endpoint_url="https://b"),
            MCPServerConfig(name="off", enabled=False, endpoint_url="https://c"),
        ])
        self.assertEqual(tools, [good_tool])
        self.assertEqual(mock_load.call_count, 2)


class TestTokenInfo(unittest.TestCase):
    def setUp(self) -> None:
        drain_token_info_events()

    def test_string_result(self) -> None:
        content = f'{_TOKEN_INFO_PREFIX}{{"token_type": "obo"}}\nreal output'
        self.assertEqual(extract_token_info_from_tool_result(content), "real output")
        self.assertEqual(drain_token_info_events(), [{"token_type": "obo"}])

    def test_block_result(self) -> None:
        content = [{"type": "text", "text": f'{_TOKEN_INFO_PREFIX}{{"a": 1}}'}, {"type": "text", "text": "out"}]
        self.assertEqual(extract_token_info_from_tool_result(content), [{"type": "text", "text": "out"}])
        self.assertEqual(drain_token_info_events(), [{"a": 1}])

    def test_no_marker_unchanged(self) -> None:
        content = [{"type": "text", "text": "out"}]
        self.assertIs(extract_token_info_from_tool_result(content), content)
        self.assertEqual(extract_token_info_from_tool_result("plain"), "plain")
        self.assertEqual(drain_token_info_events(), [])

    def test_bad_payload_is_dropped(self) -> None:
        self.assertEqual(extract_token_info_from_tool_result(f"{_TOKEN_INFO_PREFIX}not-json\nok"), "ok")
        self.assertEqual(drain_token_info_events(), [])


if __name__ == "__main__":
    unittest.main()
