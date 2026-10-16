# LangGraph Agent for AgentCore Runtime

Pre-built agent code for deployment on Amazon Bedrock AgentCore Runtime, built on [LangGraph](https://langchain-ai.github.io/langgraph/). This is an alternate custom-code framework to [`strands_agent`](../strands_agent/README.md) and [`adk_agent`](../adk_agent/README.md), selectable per agent via `agent_framework="langgraph"` at deploy time.

## Overview

This agent is configured entirely via environment variables and/or a JSON configuration file. No user-authored code is generated. The configuration schema, AgentCore Runtime entrypoint contract (streaming SSE event shapes), and telemetry span structure are kept identical to `strands_agent` so that the rest of Loom (backend, frontend, traces UI) behaves the same regardless of which framework an agent uses.

## Directory Structure

```
langgraph_agent/
├── etc/
│   ├── config.json         # Local development configuration
│   └── environment.sh      # Local env vars (gitignored)
├── src/
│   ├── __init__.py
│   ├── handler.py          # AgentCore Runtime entry point (BedrockAgentCoreApp)
│   ├── agent.py            # StateGraph construction and model selection
│   ├── config.py           # Configuration loading and validation (identical schema to strands_agent)
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── mcp_client.py         # MCP tools (langchain-mcp-adapters, OAuth2/OBO and API key httpx.Auth)
│   │   ├── a2a_client.py         # A2A agents as tools (JSON-RPC message/send)
│   │   ├── memory.py             # AgentCore Memory load/save (graph nodes)
│   │   ├── approval.py           # HITL approval policies via LangGraph interrupt()
│   │   ├── code_interpreter.py   # AgentCore Code Interpreter tools
│   │   └── secrets.py            # Secrets Manager resolution
│   └── telemetry.py        # OTEL spans (LangChain callback handler)
├── tests/
├── makefile
├── requirements.txt
└── README.md
```

## Configuration

Configuration is identical to `strands_agent`; see [`agents/strands_agent/README.md`](../strands_agent/README.md#configuration) for the full schema and environment variable reference. The agent reads configuration from `AGENT_CONFIG_JSON` (inline) or `AGENT_CONFIG_PATH` (file), and the system prompt resolves from `AGENT_SYSTEM_PROMPT` (highest priority) or the config file's `system_prompt` field.

## Development

### Setup

```bash
make install
```

### Run Locally

```bash
source etc/environment.sh
make run
```

### Run Tests

```bash
make test
```

### Build Deployment Artifact

```bash
make build
```

Produces `build/agent.zip`, a self-contained zip deployable to AgentCore Runtime. The backend's `build_agent_artifact(region, agent_framework="langgraph")` builds the same artifact from `src/` and `requirements.txt`, and the runtime uses the same `opentelemetry-instrument src/handler.py` entry point as the other frameworks.

## Architecture

The agent is a `StateGraph` compiled with an in-memory checkpointer:

```
START -> load_memory -> agent <-> tools
                          |
                          v
                     save_memory -> END
```

Each invocation runs with `thread_id` set to the Loom `session_id`, so conversation history and any paused approval persist across invocations on the same container (the equivalent of Strands' module-level agent and ADK's `InMemorySessionService`).

## Integrations

### Model Providers

`build_model()` in `src/agent.py` dispatches on `provider`:

| provider | class | auth |
|---|---|---|
| `bedrock` (default) | `ChatBedrockConverse` | IAM, no API key |
| `openai` | `ChatOpenAI` | API key from `api_key_secret_arn` |
| `anthropic` | `ChatAnthropic` | API key from `api_key_secret_arn` |
| `litellm` | `ChatOpenAI` pointed at `base_url` | proxy key from `api_key_secret_arn` |

A LiteLLM proxy exposes an OpenAI-compatible API, so `litellm` always goes through the configured proxy (`base_url` is required). `LOOM_MODEL_REQUEST_TIMEOUT_SECONDS` bounds API-key provider requests, same as Strands and ADK. A per-invocation `model_id` override is resolved and cached without rebuilding the graph.

### MCP Tool Servers

MCP tools are loaded with `langchain-mcp-adapters` (`src/integrations/mcp_client.py`). OAuth2 (M2M or OBO) and API-key auth are injected through an `httpx.Auth` on the streamable HTTP transport, reusing the same AgentCore Identity token-exchange and cache logic as `strands_agent`. Servers marked `dynamic_only` are not loaded at startup. Tools are built on the first invocation so the workload access token is available for authenticated servers.

`__TOKEN_INFO__` markers in tool results are stripped before the model sees them and surfaced as `{"token_info": ...}` stream events.

### A2A Agent Clients

Each enabled remote agent becomes a tool that takes a `message` and calls the agent's A2A JSON-RPC `message/send` method (`src/integrations/a2a_client.py`), keeping the returned `contextId` for follow-up calls. Agent cards get the same fix-ups as Strands and ADK: Salesforce's `/v1/card` path, and the configured endpoint replacing the internal URL AgentCore-hosted agents advertise. OAuth2 endpoints use a bearer-token `httpx.Auth`.

### AgentCore Memory

`load_memory` retrieves records for the latest user message into graph state (`memory`), and `save_memory` creates events for the user and assistant messages added during the invocation (`src/integrations/memory.py`). The same `LOOM_MEMORY_TELEMETRY: retrievals=N, events_sent=M` log line is emitted for cost tracking.

### Code Interpreter

Same lean wrapper over `bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter` as `adk_agent`, exposing the same six tools (execute_code, execute_command, write_files, read_files, list_files, remove_files) as LangChain tools. The session is pre-warmed in a background thread at startup.

### Human-in-the-Loop / Approval Policies

The `tools` node checks each tool call against Loom's approval policies (same env var, payload field and matching rules as Strands). A matching call pauses the graph with LangGraph's native `interrupt()`, which the handler emits as the usual `{"interrupt": ...}` event using LangGraph's interrupt id. The next invocation's `interruptResponse` resumes it with `Command(resume=...)`: `y`/`yes`/`approved` run the tool, `t` also trusts the tool for the rest of the session, anything else returns a denial to the model.

### Observability (OTEL)

`trace_invocation` wraps each invocation in an `agent.invocation` span, and `TelemetryCallbackHandler` (`src/telemetry.py`) adds `tool.call` (with `tool.name`) and `model.call` spans, keyed by LangChain run id so parallel tool calls close their own spans. ADOT's LangChain instrumentation also applies under `opentelemetry-instrument`, so the spans carry a scope AgentCore evaluations accept.

## Known Parity Gaps

Same gaps as `adk_agent`:

- MCP elicitation (`{"elicitation": ...}`, HITL Methods 3 and 4) is not wired.
- Per-invocation `dynamic_mcp_servers` from the payload are not attached.
- The WebSocket handler supports the prompt/result/error protocol only; a tool needing approval over WebSocket returns an error pointing the client to the HTTP `interruptResponse` flow.
