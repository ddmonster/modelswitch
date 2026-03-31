# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ModelSwitch is an LLM gateway proxy that exposes OpenAI-compatible and Anthropic-compatible APIs. It routes requests to upstream providers (DashScope, GLM Plan, OpenAI, Anthropic) with automatic fallback chains, per-key rate limiting, usage tracking, tool use conversion, and a web management UI.

## Running the Server

```bash
# Start server (default port 8000)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Docker
docker-compose up --build
```

The server takes ~10s to start due to provider connection warmup. Health check: `GET /api/config/health`.

## Key Commands

```bash
# Install dependencies (litellm is pinned to 1.82.6 — do not upgrade due to supply chain attack on 1.82.7/1.82.8)
pip install -r requirements.txt

# Run all tests (pytest-asyncio with strict mode)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_message_converter.py -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=term-missing

# Test an endpoint
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-gateway-admin" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm5","messages":[{"role":"user","content":"hi"}],"max_tokens":100}'
```

## Architecture

### Request Flow

```
Client → GatewayMiddleware (auth/rate-limit) → Route Handler → ChainRouter → LiteLLMAdapter → Upstream Provider
```

### Config-Driven Design (`config.yaml`)

- **Providers** are top-level connection definitions (name, base_url, api_key, protocol type). API keys support `${ENV_VAR}` substitution.
- **Models** reference providers by name with priority/timeout. Two modes:
  - `chain`: tries adapters by priority, falls back on failure (with circuit breaker + 1 retry per adapter)
  - `adapter`: direct call to a single provider, no fallback
- **API Keys** are configured in the YAML, not a database. They gate access to `/v1/*` endpoints.

### Core Layer (`app/core/`)

| Component | Purpose |
|---|---|
| `chain_router.py` | Routes requests to adapters. Chain mode does first-chunk probe for streaming fallback. |
| `circuit_breaker.py` | Per-provider circuit breaker: 5 failures → 30s open → half-open probe |
| `middleware.py` | Pure ASGI middleware (NOT BaseHTTPMiddleware — that causes infinite recursion). Auth, rate limiting, CORS. |
| `config_watcher.py` | watchdog-based hot reload with 2s debounce |

### Middleware Constraint

`GatewayMiddleware` is a **pure ASGI middleware** (raw `__call__(scope, receive, send)`), registered via `app.add_middleware()`. It does **NOT** extend `BaseHTTPMiddleware` and must not be wrapped with `@app.middleware("http")`. Mixing these two patterns causes infinite recursion because `call_next` re-triggers the decorator wrapper.

### Adapter Layer (`app/adapters/`)

`LiteLLMAdapter` wraps `litellm.acompletion()`. Model names are prefixed with the provider type: `openai/glm-5`, `anthropic/claude-sonnet-4-20250514`. Returns `AdapterResponse` dataclass with `body` (non-stream) or `stream` (async generator).

### Protocol Conversion (`app/utils/message_converter.py`)

Bidirectional conversion between Anthropic and OpenAI formats:
- `anthropic_to_openai_messages()`: System, messages, tools/tool_choice, tool_use blocks → tool_calls, tool_result blocks → role:"tool"
- `openai_stream_to_anthropic()`: OpenAI SSE chunks → Anthropic SSE events, including tool_calls deltas → input_json_delta events

### API Routes (`app/api/`)

- `openai_routes.py`: `GET /openai/models` (also `/v1/models`), `POST /openai/chat/completions` (also `/v1/chat/completions`)
- `anthropic_routes.py`: `POST /anthropic/messages` (also `/v1/messages`) — converts to/from OpenAI internally, including full tool_use conversion
- `config_routes.py`, `api_key_routes.py`: CRUD for providers/models/keys, writes back to `config.yaml`
- `usage_routes.py`: Aggregated stats with `group_by` (provider/model/api_key) and drill-down
- `log_routes.py`: Queries in-memory ring buffer (max 1000 entries)

### Protocol Conversion — Full details in `app/utils/message_converter.py`

- **Request**: `anthropic_to_openai_messages()` — converts system, messages, tools, tool_choice, tool_use/tool_result blocks
- **Non-stream response**: `_convert_openai_to_anthropic_response()` in `anthropic_routes.py` — converts tool_calls to tool_use content blocks
- **Stream response**: `openai_stream_to_anthropic()` — handles text deltas and tool_calls deltas with multi-tool index tracking

### Request Tracking (`app/utils/tracking.py`)

Centralized `track_request()` called from both route files after chain_router returns. Records usage stats (via `usage_tracker.record()`) and debug logs (via `add_log_to_buffer()`).

### Persistence

- Config: `config.yaml` (hot-reloaded via watchdog)
- Usage stats: SQLite at `data/usage.db`, batch-flushed every 10s
- Logs: Rotating file at `logs/gateway.log` + in-memory deque buffer

### Frontend

Single-page app in `web/` (HTML/CSS/JS, no build step). 5 tabs: Providers, Models, API Keys, Usage Stats, Debug Logs. Served at `/` and `/web/`.

## Key Patterns

- Route handlers access shared state via `request.app.state` (chain_router, usage_tracker, api_key_service, config)
- The middleware injects auth info into `scope["state"]` (api_key, api_key_name, api_key_config), which maps to `request.state` in route handlers
- Public paths (no auth required): `/`, `/health`, `/metrics`, `/docs`, `/web/*`, `/api/*`
- Auth-required paths: `/v1/*`, `/openai/*`, `/anthropic/*` — accepts `Authorization: Bearer <key>`, `x-api-key: <key>`, or bare `sk-*` header
- Error format adapts to route: OpenAI-style `{"error": {...}}` for `/openai/*` and `/v1/chat/completions`, Anthropic-style `{"type": "error", "error": {...}}` for `/anthropic/*` and `/v1/messages`
- Anthropic routes skip conversion when the first adapter is an Anthropic provider (passthrough mode)
