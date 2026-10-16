"""AgentCore Memory integration for the LangGraph agent.

Ports the semantics of ``agents/strands_agent/src/integrations/memory.py``
(same ``MEMORY_STORE_ID`` env var, same namespace/actor-id defaults, same
``LOOM_MEMORY_TELEMETRY`` structured log line) onto two LangGraph nodes:

- ``load_memory`` runs first on every invocation (the analog of Strands'
  ``BeforeInvocationEvent``/ADK's ``before_agent_callback``). It retrieves
  memory records for the latest user message and stores them in graph state
  under ``memory``, and records how many messages existed before this turn.
- ``save_memory`` runs last (the analog of ``AfterInvocationEvent``). It
  saves only the messages added during this invocation, exactly like
  Strands snapshots ``len(agent.messages)``.

``MemoryManager`` holds the boto3 client and counters; the nodes in
``agent.py`` call its ``load``/``save`` methods.
"""

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import boto3
from botocore.exceptions import ClientError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

# Default namespace used for memory record retrieval and event creation
_DEFAULT_NAMESPACE = "loom"
# Default actor ID for events created by the agent
_DEFAULT_ACTOR_ID = "loom-agent"

# AWS actorId constraint: [a-zA-Z0-9][a-zA-Z0-9-_/]*(?::[a-zA-Z0-9-_/]+)*[a-zA-Z0-9-_/]*
_ACTOR_ID_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9\-_/:]")


def _sanitize_actor_id(raw: str) -> str:
    """Sanitize actor_id to comply with AWS actorId constraints as a fallback."""
    sanitized = _ACTOR_ID_INVALID_CHARS.sub("-", raw)
    if not sanitized or not sanitized[0].isalnum():
        sanitized = "u-" + sanitized
    return sanitized


def message_text(message: BaseMessage) -> str:
    """Return the plain text of a LangChain message (string or text blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    texts = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            texts.append(block["text"])
    return "".join(texts)


class MemoryManager:
    """Loads and saves conversation context via AgentCore Memory.

    If no memory store id is configured the manager is a no-op, so callers
    can wire it in regardless of whether memory is enabled.
    """

    def __init__(self, memory_store_id: str | None = None) -> None:
        store_id = memory_store_id or os.environ.get("MEMORY_STORE_ID", "")
        self.memory_store_id: str | None = store_id if store_id else None
        self._client: Any = None
        self.retrievals: int = 0
        self.events_sent: int = 0

        if not self.memory_store_id:
            logger.info("MEMORY_STORE_ID not set; MemoryManager is disabled")

    @property
    def client(self) -> Any:
        """Lazily initialize the bedrock-agentcore client."""
        if self._client is None and self.memory_store_id:
            self._client = boto3.client(
                "bedrock-agentcore",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )
        return self._client

    async def load(self, messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
        """Retrieve memory records relevant to the latest user message."""
        self.retrievals = 0
        self.events_sent = 0
        if not self.memory_store_id:
            return []

        logger.info("MemoryManager: before_invocation — retrieving from store '%s'", self.memory_store_id)
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = message_text(msg)
                if query:
                    break
        if not query:
            logger.info("MemoryManager: no query text found in messages; skipping memory retrieval")
            return []

        try:
            response = await asyncio.to_thread(
                self.client.retrieve_memory_records,
                memoryId=self.memory_store_id,
                namespace=_DEFAULT_NAMESPACE,
                searchCriteria={"searchQuery": query},
            )
            records = response.get("memoryRecordSummaries", [])
            self.retrievals = len(records)
            logger.info(
                "MemoryManager: retrieved %d memory record(s) from store '%s'",
                len(records),
                self.memory_store_id,
            )
            return records
        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDeniedException":
                logger.warning("Access denied retrieving memory from store '%s': %s", self.memory_store_id, e.response["Error"]["Message"])
            else:
                logger.exception("Failed to retrieve memory from store '%s'", self.memory_store_id)
        except Exception:
            logger.exception("Failed to retrieve memory from store '%s'", self.memory_store_id)
        return []

    async def save(
        self,
        new_messages: Sequence[BaseMessage],
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Save the messages added during this invocation."""
        if not self.memory_store_id:
            self._emit_telemetry()
            return

        logger.info("MemoryManager: after_invocation — saving to store '%s'", self.memory_store_id)
        try:
            if not new_messages:
                logger.info("MemoryManager: no new messages to save to memory store")
                return

            now = datetime.now(timezone.utc)
            safe_actor_id = _sanitize_actor_id(actor_id or _DEFAULT_ACTOR_ID)
            for msg in new_messages:
                if isinstance(msg, HumanMessage):
                    ac_role = "USER"
                elif isinstance(msg, AIMessage):
                    ac_role = "ASSISTANT"
                else:
                    # Tool results and system messages are not conversational turns.
                    continue

                text = message_text(msg)
                if not text:
                    continue

                create_kwargs: dict[str, Any] = {
                    "memoryId": self.memory_store_id,
                    "actorId": safe_actor_id,
                    "eventTimestamp": now,
                    "payload": [{"conversational": {"content": {"text": text}, "role": ac_role}}],
                }
                if session_id:
                    create_kwargs["sessionId"] = session_id

                await asyncio.to_thread(self.client.create_event, **create_kwargs)
                self.events_sent += 1

            logger.info(
                "MemoryManager: created %d event(s) in memory store '%s'",
                self.events_sent,
                self.memory_store_id,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDeniedException":
                logger.warning("Access denied saving memory to store '%s': %s", self.memory_store_id, e.response["Error"]["Message"])
            else:
                logger.exception("Failed to save memory to store '%s'", self.memory_store_id)
        except Exception:
            logger.exception("Failed to save memory to store '%s'", self.memory_store_id)
        finally:
            self._emit_telemetry()

    def _emit_telemetry(self) -> None:
        """Emit a structured log line with memory usage counters for cost tracking."""
        logger.info(
            "LOOM_MEMORY_TELEMETRY: retrievals=%d, events_sent=%d",
            self.retrievals,
            self.events_sent,
        )
