"""Tests for the AgentCore Memory integration."""

import os
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.integrations.memory import MemoryManager, _sanitize_actor_id, message_text


class TestHelpers(unittest.TestCase):
    def test_sanitize_actor_id(self) -> None:
        self.assertEqual(_sanitize_actor_id("alice@example.com"), "alice-example-com")
        self.assertEqual(_sanitize_actor_id("-bob"), "u--bob")

    def test_message_text_blocks(self) -> None:
        msg = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "tool_use", "id": "1"}, {"type": "text", "text": "b"}])
        self.assertEqual(message_text(msg), "ab")


class TestMemoryManager(unittest.IsolatedAsyncioTestCase):
    def _manager(self) -> tuple[MemoryManager, MagicMock]:
        manager = MemoryManager(memory_store_id="mem-1")
        client = MagicMock()
        manager._client = client
        return manager, client

    async def test_disabled_without_store_id(self) -> None:
        with patch.dict(os.environ, {"MEMORY_STORE_ID": ""}):
            manager = MemoryManager()
        self.assertIsNone(manager.memory_store_id)
        self.assertEqual(await manager.load([HumanMessage(content="x")]), [])
        await manager.save([HumanMessage(content="x")])

    async def test_store_id_from_env(self) -> None:
        with patch.dict(os.environ, {"MEMORY_STORE_ID": "env-mem"}):
            self.assertEqual(MemoryManager().memory_store_id, "env-mem")

    async def test_load_queries_latest_user_message(self) -> None:
        manager, client = self._manager()
        client.retrieve_memory_records.return_value = {"memoryRecordSummaries": [{"id": 1}, {"id": 2}]}
        records = await manager.load([HumanMessage(content="old"), AIMessage(content="r"), HumanMessage(content="new")])
        self.assertEqual(len(records), 2)
        self.assertEqual(manager.retrievals, 2)
        kwargs = client.retrieve_memory_records.call_args.kwargs
        self.assertEqual(kwargs["memoryId"], "mem-1")
        self.assertEqual(kwargs["namespace"], "loom")
        self.assertEqual(kwargs["searchCriteria"], {"searchQuery": "new"})

    async def test_load_skips_without_query(self) -> None:
        manager, client = self._manager()
        self.assertEqual(await manager.load([AIMessage(content="r")]), [])
        client.retrieve_memory_records.assert_not_called()

    async def test_load_swallows_access_denied(self) -> None:
        manager, client = self._manager()
        client.retrieve_memory_records.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "RetrieveMemoryRecords"
        )
        self.assertEqual(await manager.load([HumanMessage(content="q")]), [])

    async def test_save_maps_roles_and_skips_tool_messages(self) -> None:
        manager, client = self._manager()
        await manager.save(
            [HumanMessage(content="hi"), AIMessage(content=""), ToolMessage(content="r", tool_call_id="1"), AIMessage(content="hello")],
            actor_id="user@x.com",
            session_id="s1",
        )
        self.assertEqual(client.create_event.call_count, 2)
        first, second = (c.kwargs for c in client.create_event.call_args_list)
        self.assertEqual(first["payload"][0]["conversational"], {"content": {"text": "hi"}, "role": "USER"})
        self.assertEqual(second["payload"][0]["conversational"]["role"], "ASSISTANT")
        self.assertEqual(first["actorId"], "user-x-com")
        self.assertEqual(first["sessionId"], "s1")
        self.assertEqual(manager.events_sent, 2)

    async def test_save_emits_telemetry_line(self) -> None:
        manager, _client = self._manager()
        with self.assertLogs("src.integrations.memory", level="INFO") as logs:
            await manager.save([HumanMessage(content="hi")])
        self.assertTrue(any("LOOM_MEMORY_TELEMETRY: retrievals=0, events_sent=1" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
