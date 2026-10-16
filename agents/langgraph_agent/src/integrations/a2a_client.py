"""A2A (Agent-to-Agent) tools from agent configuration.

Each enabled remote agent becomes one LangChain ``StructuredTool`` that the
model can call with a natural-language ``message``. The tool speaks the A2A
JSON-RPC ``message/send`` method directly over ``httpx``, which keeps this
package free of ``a2a-sdk`` (whose types changed shape between the
generation Strands uses and the one ADK pulls in).

Ports the same auth handling and remote-card quirks as the Strands and ADK
implementations:

- OAuth2 (M2M or OBO) bearer tokens via the shared ``_OAuth2TokenFetcher``
  from ``mcp_client.py``, injected with an ``httpx.Auth`` so the card fetch
  and every message send carry the same token.
- Salesforce Einstein AI Agent endpoints expose their card at ``/v1/card``.
- AgentCore-hosted agents advertise an internal URL (e.g.
  ``http://localhost:8081``) in their card; the configured endpoint is used
  instead. Non-AgentCore agents keep their advertised URL.
"""

import logging
import re
import uuid
from typing import Any, Generator, Optional

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.config import A2AAgentConfig
from src.integrations.mcp_client import _OAuth2TokenFetcher

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 300
_AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"


class _A2AToolInput(BaseModel):
    message: str = Field(description="The message to send to the remote agent.")


def _sanitize_tool_name(raw: str) -> str:
    """Turn a configured agent name into a tool name every provider accepts."""
    sanitized = re.sub(r"[^0-9a-zA-Z_-]", "_", raw)
    if not sanitized or not sanitized[0].isalpha():
        sanitized = f"a2a_{sanitized}"
    return sanitized[:64]


class _BearerAuth(httpx.Auth):
    """``httpx.Auth`` that injects a Bearer token fetched via ``_OAuth2TokenFetcher``."""

    def __init__(self, fetcher: _OAuth2TokenFetcher) -> None:
        self._fetcher = fetcher

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        token = self._fetcher.fetch_bearer_token()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request


def _build_http_client(config: A2AAgentConfig) -> httpx.AsyncClient:
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
            "A2A agent '%s' configured with OAuth2 auth (credential_provider=%s, scopes=%s, delegation_mode=%s)",
            config.name, config.auth.credential_provider_name, scope_list, delegation_mode,
        )
        return httpx.AsyncClient(auth=_BearerAuth(fetcher), timeout=_DEFAULT_TIMEOUT)
    return httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)


async def fetch_agent_card(endpoint_url: str, http_client: httpx.AsyncClient) -> tuple[dict[str, Any], str]:
    """Fetch a remote agent's card and resolve the URL to send messages to.

    Returns ``(card, rpc_url)``.
    """
    base_url = endpoint_url.rstrip("/")
    is_salesforce = "salesforce.com/einstein/ai-agent" in base_url
    card_url = f"{base_url}/v1/card" if is_salesforce else f"{base_url}{_AGENT_CARD_WELL_KNOWN_PATH}"

    response = await http_client.get(card_url)
    response.raise_for_status()
    card: dict[str, Any] = response.json()
    logger.info("Fetched agent card from %s (name=%s)", card_url, card.get("name"))

    if is_salesforce:
        rpc_url = card.get("url") or base_url
        logger.info("Using agent card url as-is (endpoint='%s')", endpoint_url)
    else:
        rpc_url = base_url + "/"
        logger.info("Overriding agent card url '%s' -> '%s'", card.get("url"), rpc_url)
    return card, rpc_url


def _parts_text(parts: list[dict[str, Any]] | None) -> str:
    texts = []
    for part in parts or []:
        if part.get("kind", part.get("type")) == "text" and part.get("text"):
            texts.append(part["text"])
    return "\n".join(texts)


def extract_result_text(result: dict[str, Any]) -> str:
    """Extract the reply text from an A2A ``message/send`` result.

    The result is either a ``Message`` (``parts``) or a ``Task`` (text in
    ``artifacts[].parts`` and/or ``status.message.parts``).
    """
    if result.get("kind") == "message" or "parts" in result:
        return _parts_text(result.get("parts"))
    texts = [_parts_text(a.get("parts")) for a in result.get("artifacts") or []]
    texts = [t for t in texts if t]
    if texts:
        return "\n".join(texts)
    status_message = (result.get("status") or {}).get("message") or {}
    return _parts_text(status_message.get("parts"))


class A2AAgentClient:
    """Sends messages to one remote A2A agent, keeping its ``contextId`` so
    follow-up calls continue the same remote conversation.
    """

    def __init__(self, config: A2AAgentConfig, http_client: Optional[httpx.AsyncClient] = None) -> None:
        self.config = config
        self.http_client = http_client or _build_http_client(config)
        self.card: dict[str, Any] = {}
        self.rpc_url = ""
        self._context_id: Optional[str] = None

    async def resolve(self) -> None:
        self.card, self.rpc_url = await fetch_agent_card(self.config.endpoint_url, self.http_client)

    async def send(self, message: str) -> str:
        msg: dict[str, Any] = {
            "role": "user",
            "kind": "message",
            "messageId": uuid.uuid4().hex,
            "parts": [{"kind": "text", "text": message}],
        }
        if self._context_id:
            msg["contextId"] = self._context_id
        body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/send",
            "params": {"message": msg},
        }
        response = await self.http_client.post(self.rpc_url, json=body)
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            err = data["error"]
            return f"Error from A2A agent '{self.config.name}': {err.get('message', err)}"
        result = data.get("result") or {}
        self._context_id = result.get("contextId") or self._context_id
        return extract_result_text(result) or "(the remote agent returned no text)"


async def _build_a2a_tool(config: A2AAgentConfig) -> BaseTool:
    client = A2AAgentClient(config)
    await client.resolve()
    description = client.card.get("description") or f"Send a message to the '{config.name}' A2A agent."

    async def _call(message: str) -> str:
        return await client.send(message)

    logger.info("Created A2A tool for agent '%s' at %s", config.name, config.endpoint_url)
    return StructuredTool.from_function(
        coroutine=_call,
        name=_sanitize_tool_name(config.name),
        description=description,
        args_schema=_A2AToolInput,
    )


async def create_a2a_tools(agents: list[A2AAgentConfig]) -> list[BaseTool]:
    """Create one tool per enabled, reachable A2A agent.

    Agents whose card can't be resolved (unreachable endpoint, auth
    failure) are skipped so the rest of the agent's tools still load.
    """
    tools: list[BaseTool] = []
    for agent_cfg in agents:
        if not agent_cfg.enabled:
            logger.debug("Skipping disabled A2A agent '%s'", agent_cfg.name)
            continue
        try:
            tools.append(await _build_a2a_tool(agent_cfg))
        except Exception as e:
            logger.warning(
                "Failed to build A2A tool for agent '%s': %s. Skipping this agent.",
                agent_cfg.name, e,
            )
    logger.info("Initialised %d A2A tool(s)", len(tools))
    return tools
