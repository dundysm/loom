"""MCP tool loading from agent configuration.

Ports the OAuth2/OBO token-exchange and API-key auth logic from
``agents/strands_agent/src/integrations/mcp_client.py`` (same token cache,
same JWT-claims decode, same AgentCore Identity API calls). Like Strands,
auth is injected through an ``httpx.Auth`` on the streamable HTTP transport;
``langchain-mcp-adapters`` accepts that directly via the connection's
``auth`` field, so a fresh bearer token is resolved for every MCP request.

Tools are loaded with ``load_mcp_tools(None, connection=...)``, which opens
a short-lived MCP session per tool call. That keeps tool calls safe to run
from any invocation (each request carries the current workload token) at
the cost of a session handshake per call.

Known parity gap: MCP elicitation (Strands' ``elicitation_callback``) and
out-of-band ``logging/message`` token_info notifications are not wired.
Only the result-embedded ``__TOKEN_INFO__`` marker path
(``extract_token_info_from_tool_result``) is supported, same as ADK.
"""

import base64
import hashlib
import json as _json
import logging
import os
import threading
import time
from typing import Any, Generator, Optional

import boto3
import httpx
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_mcp_adapters.tools import load_mcp_tools

from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from bedrock_agentcore.services.identity import IdentityClient

from src.config import MCPServerConfig

logger = logging.getLogger(__name__)


# Process-wide cache for downstream tokens, keyed by
# (credential_provider_name, oauth2_flow, workload_token_prefix).
_TOKEN_CACHE: dict[tuple[str, str, str], tuple[str, float]] = {}
_TOKEN_CACHE_LOCK = threading.Lock()
_TOKEN_EXPIRY_SKEW_SECS = 30

# User access token for OBO flows — set per invocation from the Lambda payload
_user_access_token: str | None = None
_user_access_token_lock = threading.Lock()


def set_user_access_token(token: str | None) -> None:
    """Set the user access token for OBO token exchange flows."""
    global _user_access_token
    with _user_access_token_lock:
        _user_access_token = token


def get_user_access_token() -> str | None:
    """Get the current user access token."""
    with _user_access_token_lock:
        return _user_access_token


# Token info events emitted when OBO tokens are acquired.
# The handler drains this list and yields events to the stream.
_token_info_events: list[dict[str, Any]] = []
_token_info_emitted: set[str] = set()
_token_info_lock = threading.Lock()


def drain_token_info_events() -> list[dict[str, Any]]:
    """Drain and return all pending token info events."""
    with _token_info_lock:
        events = _token_info_events.copy()
        _token_info_events.clear()
    return events


def reset_token_info_state() -> None:
    """Reset emission tracking for a new invocation."""
    with _token_info_lock:
        _token_info_emitted.clear()


_TOKEN_INFO_PREFIX = "__TOKEN_INFO__:"  # nosec B105 — protocol prefix string, not a password


def _split_token_info(text: str) -> tuple[str, bool]:
    """Remove ``__TOKEN_INFO__`` lines from ``text``, queueing their payloads."""
    kept: list[str] = []
    found = False
    for line in text.splitlines(keepends=True):
        if line.startswith(_TOKEN_INFO_PREFIX):
            found = True
            try:
                payload = _json.loads(line[len(_TOKEN_INFO_PREFIX):])
                with _token_info_lock:
                    _token_info_events.append(payload)
                logger.info(
                    "Extracted token_info from tool result: type=%s provider=%s",
                    payload.get("token_type"),
                    payload.get("credential_provider"),
                )
            except Exception as e:
                logger.warning("Failed to parse __TOKEN_INFO__ block: %s", e)
        else:
            kept.append(line)
    return "".join(kept), found


def extract_token_info_from_tool_result(content: Any) -> Any:
    """Strip ``__TOKEN_INFO__`` markers from a LangChain tool result.

    Ports Strands' ``TokenInfoHook._extract_token_info``. MCP tools loaded by
    ``langchain-mcp-adapters`` return either a plain string or a list of
    content blocks (``{"type": "text", "text": ...}``); both are handled.
    Called by the tools node in ``agent.py`` on every tool result, and the
    handler drains the queued events into ``{"token_info": ...}`` stream
    events.
    """
    if isinstance(content, str):
        cleaned, found = _split_token_info(content)
        return cleaned if found else content
    if isinstance(content, list):
        clean_blocks: list[Any] = []
        changed = False
        for block in content:
            text = block.get("text", "") if isinstance(block, dict) else (block if isinstance(block, str) else "")
            if text and text.startswith(_TOKEN_INFO_PREFIX):
                _split_token_info(text)
                changed = True
                continue
            clean_blocks.append(block)
        return clean_blocks if changed else content
    return content


def _token_fingerprint(token: str) -> str:
    """Non-reversible identifier for a token, safe to log (unlike a raw prefix)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _decode_jwt_claims(token: str) -> dict[str, Any] | None:
    """Decode JWT payload without verification (for inspection only)."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        return _json.loads(base64.urlsafe_b64decode(padded))
    except Exception:
        return None


class _OAuth2TokenFetcher:
    """Fetches downstream OAuth2 access tokens via the AgentCore Identity
    service, shared by the MCP ``httpx.Auth`` handler (this module) and the A2A
    ``httpx.Auth`` handler (``a2a_client.py``) — the token-fetch mechanics are
    identical for both; only the injection point (HTTP header vs httpx auth
    flow) differs by transport.

    On each call the fetcher:
      1. Uses the workload access token (resolved from
         ``BedrockAgentCoreContext``, a ContextVar set by the AgentCore
         Runtime request handler independent of any agent framework).
      2. Exchanges it for a downstream OAuth2 access token via the
         AgentCore ``get_resource_oauth2_token`` API.

    The delegation_mode determines the oauth2Flow parameter:
      - "m2m" → M2M (machine-to-machine)
      - "obo" → ON_BEHALF_OF_TOKEN_EXCHANGE (on-behalf-of user)
    """

    def __init__(
        self,
        credential_provider_name: str,
        scopes: list[str],
        delegation_mode: str = "m2m",
        obo_grant_type: str | None = None,
        audience: str = "",
    ) -> None:
        self._credential_provider_name = credential_provider_name
        self._scopes = scopes
        self._region = os.environ.get("AWS_REGION", "us-east-1")
        self._delegation_mode = (delegation_mode or "m2m").lower()
        self._oauth2_flow = "ON_BEHALF_OF_TOKEN_EXCHANGE" if self._delegation_mode == "obo" else "M2M"
        self._obo_grant_type = obo_grant_type
        self._audience = audience

    def fetch_bearer_token(self) -> Optional[str]:
        """Return a bearer token string for the current request, or None."""
        workload_token = BedrockAgentCoreContext.get_workload_access_token()
        if not workload_token:
            logger.warning(
                "No workload access token available for '%s'; sending unauthenticated",
                self._credential_provider_name,
            )
            return None
        logger.info(
            "Workload token for '%s': fingerprint=%s len=%d",
            self._credential_provider_name, _token_fingerprint(workload_token), len(workload_token),
        )

        token = self._fetch_resource_token(workload_token)
        if token:
            logger.debug(
                "Fetched OAuth2 token for credential provider '%s' (flow=%s)",
                self._credential_provider_name, self._oauth2_flow,
            )
            return token

        logger.warning(
            "No accessToken available for credential provider '%s' (flow=%s); sending unauthenticated",
            self._credential_provider_name, self._oauth2_flow,
        )
        return None

    def _emit_token_info(self, token: str) -> None:
        """Decode token claims and emit a token_info event (once per drain cycle)."""
        emit_key = self._credential_provider_name
        with _token_info_lock:
            if emit_key in _token_info_emitted:
                return
            _token_info_emitted.add(emit_key)

        claims = _decode_jwt_claims(token)
        if claims:
            event = {
                "token_type": "obo",
                "credential_provider": self._credential_provider_name,
                "flow": self._oauth2_flow,
                "claims": {
                    "iss": claims.get("iss"),
                    "sub": claims.get("sub"),
                    "aud": claims.get("aud"),
                    "azp": claims.get("azp"),
                    "appid": claims.get("appid"),
                    "cid": claims.get("cid"),
                    "scp": claims.get("scp"),
                    "roles": claims.get("roles"),
                    "act": claims.get("act"),
                    "exp": claims.get("exp"),
                    "iat": claims.get("iat"),
                },
            }
            with _token_info_lock:
                _token_info_events.append(event)

    def _fetch_resource_token(self, workload_token: str) -> Optional[str]:
        cache_key = (self._credential_provider_name, self._oauth2_flow, workload_token[:32])
        now = time.time()
        with _TOKEN_CACHE_LOCK:
            cached = _TOKEN_CACHE.get(cache_key)
            if cached and cached[1] > now + _TOKEN_EXPIRY_SKEW_SECS:
                self._emit_token_info(cached[0])
                return cached[0]

        try:
            identity_client = IdentityClient(self._region)

            token_kwargs: dict[str, Any] = {
                'workloadIdentityToken': workload_token,
                'resourceCredentialProviderName': self._credential_provider_name,
                'scopes': self._scopes,
                'oauth2Flow': self._oauth2_flow,
            }
            if self._obo_grant_type == "JWT_AUTHORIZATION_GRANT":
                token_kwargs['customParameters'] = {'requested_token_use': 'on_behalf_of'}
            if self._obo_grant_type == "TOKEN_EXCHANGE":
                if self._audience:
                    token_kwargs['audiences'] = [self._audience]
                token_kwargs['customParameters'] = {
                    'subject_token_type': 'urn:ietf:params:oauth:token-type:access_token',
                }
            resp = identity_client.dp_client.get_resource_oauth2_token(**token_kwargs)
            token = resp.get("accessToken")
            if not token:
                logger.warning(
                    "No accessToken returned for '%s' (flow=%s)",
                    self._credential_provider_name, self._oauth2_flow,
                )
                return None

            expires_in = int(resp.get("expiresIn") or 300)
            with _TOKEN_CACHE_LOCK:
                _TOKEN_CACHE[cache_key] = (token, now + expires_in)

            logger.info(
                "OAuth2 token acquired: credential_provider=%s flow=%s expires_in=%ds",
                self._credential_provider_name, self._oauth2_flow, expires_in,
            )

            self._emit_token_info(token)
            return token
        except Exception as e:
            logger.warning(
                "OAuth2 token exchange failed: credential_provider=%s flow=%s error=%s",
                self._credential_provider_name, self._oauth2_flow, e,
            )
            return None


class _OAuth2Auth(httpx.Auth):
    """``httpx.Auth`` that injects ``Authorization: Bearer <token>`` fetched
    via ``_OAuth2TokenFetcher`` on every request.
    """

    def __init__(self, fetcher: _OAuth2TokenFetcher) -> None:
        self._fetcher = fetcher

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        token = self._fetcher.fetch_bearer_token()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request


class _ApiKeyAuth(httpx.Auth):
    """``httpx.Auth`` that injects an API key header on each request.

    The API key is resolved once from AWS Secrets Manager at initialization
    (not per-request) to avoid throttling.
    """

    def __init__(self, secret_name: str, header_name: str = "x-api-key") -> None:
        self._header_name = header_name
        self._api_key = self._resolve_key(secret_name)

    @staticmethod
    def _resolve_key(secret_name: str) -> str:
        region = os.environ.get("AWS_REGION", "us-east-1")
        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=secret_name)
        return resp["SecretString"]

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        if not self._api_key:
            logger.warning("No API key resolved; sending unauthenticated request")
        elif self._header_name.lower() == "authorization":
            request.headers["Authorization"] = f"Bearer {self._api_key}"
        else:
            request.headers[self._header_name] = self._api_key
        yield request


def _build_auth(config: MCPServerConfig) -> Optional[httpx.Auth]:
    """Build an ``httpx.Auth`` for the given MCP server configuration.

    Returns None for unauthenticated servers.
    """
    if config.auth and config.auth.type == "oauth2" and config.auth.credential_provider_name:
        scope_list = config.auth.scopes.split() if config.auth.scopes else []
        delegation_mode = (config.auth.delegation_mode or "m2m").lower()
        fetcher = _OAuth2TokenFetcher(
            credential_provider_name=config.auth.credential_provider_name,
            scopes=scope_list,
            delegation_mode=delegation_mode,
            obo_grant_type=config.auth.obo_grant_type or None,
            audience=config.auth.audience or "",
        )
        logger.info(
            "MCP server '%s' configured with OAuth2 auth (credential_provider=%s, scopes=%s, delegation_mode=%s, obo_grant_type=%s)",
            config.name,
            config.auth.credential_provider_name,
            scope_list,
            delegation_mode,
            config.auth.obo_grant_type,
        )
        return _OAuth2Auth(fetcher)

    if config.auth and config.auth.type == "api_key" and config.auth.credentials_secret_arn:
        try:
            auth = _ApiKeyAuth(
                secret_name=config.auth.credentials_secret_arn,
                header_name=config.auth.api_key_header_name or "x-api-key",
            )
            logger.info(
                "MCP server '%s' configured with API key auth (secret=%s, header=%s)",
                config.name,
                config.auth.credentials_secret_arn,
                config.auth.api_key_header_name,
            )
            return auth
        except Exception as e:
            logger.warning(
                "Failed to resolve API key for server '%s': %s. Falling back to unauthenticated.",
                config.name,
                e,
            )
    return None


def build_connection(server: MCPServerConfig) -> Optional[StreamableHttpConnection]:
    """Build a ``langchain-mcp-adapters`` connection for one server.

    Only the ``streamable_http`` transport is supported, matching Strands
    and ADK. Returns None for unsupported transports.
    """
    if server.transport != "streamable_http":
        logger.warning(
            "Unsupported MCP transport '%s' for server '%s'; skipping",
            server.transport,
            server.name,
        )
        return None

    connection: StreamableHttpConnection = {
        "transport": "streamable_http",
        "url": server.endpoint_url,
    }
    auth = _build_auth(server)
    if auth is not None:
        connection["auth"] = auth
    return connection


async def load_tools(servers: list[MCPServerConfig]) -> list[BaseTool]:
    """Load LangChain tools from every enabled MCP server.

    Servers that fail to connect are skipped so the rest of the agent's
    tools still load, matching Strands' and ADK's soft-fail behavior.
    """
    tools: list[BaseTool] = []
    loaded_servers = 0
    for server in servers:
        if not server.enabled:
            logger.debug("Skipping disabled MCP server '%s'", server.name)
            continue
        connection = build_connection(server)
        if connection is None:
            continue
        try:
            server_tools = await load_mcp_tools(None, connection=connection, server_name=server.name)
        except Exception as e:
            logger.warning("Failed to load tools from MCP server '%s': %s. Skipping this server.", server.name, e)
            continue
        tools.extend(server_tools)
        loaded_servers += 1
        logger.info("Loaded %d tool(s) from MCP server '%s'", len(server_tools), server.name)
    logger.info("Loaded %d MCP tool(s) from %d server(s)", len(tools), loaded_servers)
    return tools


def has_oauth2_servers(servers: list[MCPServerConfig]) -> bool:
    """Check if any enabled MCP servers require OAuth2 authentication."""
    return any(
        s.enabled and s.auth and s.auth.type == "oauth2" and s.auth.credential_provider_name
        for s in servers
    )


def has_deferred_auth_servers(servers: list[MCPServerConfig]) -> bool:
    """Check if any enabled MCP servers require deferred auth (OAuth2 or API key)."""
    return any(
        s.enabled and s.auth and s.auth.type in ("oauth2", "api_key")
        for s in servers
    )
