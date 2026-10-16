"""Test doubles shared by the LangGraph agent tests."""

import json
from typing import Any, Iterator, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class FakeToolChatModel(BaseChatModel):
    """Chat model that replays scripted ``AIMessage`` responses.

    Streams text word by word and tool calls as ``tool_call_chunks`` so the
    handler's streaming path is exercised the same way a real provider
    would exercise it.
    """

    responses: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolChatModel":
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        return self.responses.pop(0)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next(messages)
        text = message.content if isinstance(message.content, str) else ""
        words = text.split(" ") if text else []
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
        for index, tc in enumerate(message.tool_calls):
            chunk = ChatGenerationChunk(message=AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": tc["name"],
                    "args": json.dumps(tc["args"]),
                    "id": tc["id"],
                    "index": index,
                }],
            ))
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk


def tool_call_message(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])
