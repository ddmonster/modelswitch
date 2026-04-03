# ModelSwitch Test Guide

This guide covers all testing approaches for ModelSwitch - an LLM gateway proxy that exposes OpenAI-compatible and Anthropic-compatible APIs.

## Quick Start

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=term-missing

# Run smoke test against running server
python scripts/smoketest.py --model glm-5

# Run end-to-end tests
MODELSWITCH_E2E=1 pytest tests/test_e2e.py -v
```

---

## Test Categories

### 1. Unit Tests

Unit tests cover individual components in isolation. They mock external dependencies and test specific functions/classes.

**Location:** `tests/test_*.py`

| Test File | Coverage | Description |
|-----------|----------|-------------|
| `test_anthropic_adapter.py` | 100% | Anthropic SDK adapter |
| `test_openai_adapter.py` | 100% | OpenAI SDK adapter |
| `test_chain_router.py` | 100% | Request routing & fallback |
| `test_request_queue.py` | 100% | Request queue & concurrency |
| `test_circuit_breaker.py` | 98% | Circuit breaker pattern |
| `test_api_key_service.py` | 100% | API key management |
| `test_config_models.py` | 100% | Configuration models |
| `test_message_converter.py` | 96% | Protocol conversion |
| `test_usage_tracker.py` | 95% | Usage statistics |
| `test_circuit_breaker.py` | 98% | Circuit breaker |
| `test_conversation_routes.py` | 100% | Conversation API |

**Run:**
```bash
# All unit tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_chain_router.py -v

# Specific test class
python -m pytest tests/test_api_routes.py::TestOpenAIChatCompletions -v

# Specific test
python -m pytest tests/test_api_routes.py::TestOpenAIChatCompletions::test_stream_success -v
```

### 2. API Integration Tests

Tests the HTTP API layer using httpx AsyncClient with ASGI transport. These tests mock the upstream adapters but exercise the full request/response cycle.

**Location:** `tests/test_api_routes.py`

**Coverage:**
- Authentication (Bearer, x-api-key, sk-prefix)
- Rate limiting
- OpenAI chat completions (stream/non-stream)
- Anthropic messages (stream/non-stream)
- Tool/function calling
- Config CRUD operations
- Usage statistics
- Debug logs

**Run:**
```bash
python -m pytest tests/test_api_routes.py -v
```

### 3. End-to-End Tests

Tests against a running server with real HTTP requests. These exercise the full stack including network layer.

**Location:** `tests/test_e2e.py`

**Prerequisites:**
- Server must be running
- Valid API key configured
- At least one model available

**Run:**
```bash
# Start server first
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run e2e tests
MODELSWITCH_E2E=1 pytest tests/test_e2e.py -v

# With custom settings
MODELSWITCH_E2E=1 \
MODELSWITCH_URL=http://localhost:8000 \
MODELSWITCH_ADMIN_KEY=sk-gateway-admin \
MODEL_NAME=glm-5 \
pytest tests/test_e2e.py -v
```

**Test Classes:**
| Class | Tests | Description |
|-------|-------|-------------|
| `TestHealthAndConfig` | 4 | Health, models, providers, config |
| `TestAuthentication` | 4 | Auth methods, invalid keys |
| `TestOpenAIChatCompletions` | 5 | Chat completions, streaming, errors |
| `TestAnthropicMessages` | 4 | Anthropic API compatibility |
| `TestToolCalls` | 2 | Function calling |
| `TestErrorHandling` | 5 | Error responses, CORS |
| `TestStreamingBehavior` | 3 | SSE format, cancellation |
| `TestConcurrency` | 2 | Concurrent requests |

### 4. Smoke Tests

Quick validation tests for manual testing or CI/CD health checks.

**Location:** `scripts/smoketest.py` (Python), `scripts/smoketest.sh` (Bash)

**Run:**
```bash
# Python version
python scripts/smoketest.py --model glm-5

# Quick mode (skip streaming)
python scripts/smoketest.py --model glm-5 --quick

# Custom server
python scripts/smoketest.py --url http://localhost:8000 --key sk-gateway-admin --model glm-5

# Bash version
./scripts/smoketest.sh
MODELSWITCH_URL=http://localhost:8000 ./scripts/smoketest.sh
```

---

## Environment Variables

### Test Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELSWITCH_E2E` | (unset) | Set to `1` to enable e2e tests |
| `MODELSWITCH_URL` | `http://localhost:8000` | Server URL for e2e/smoke tests |
| `MODELSWITCH_ADMIN_KEY` | `sk-gateway-admin` | Admin API key |
| `MODEL_NAME` | `glm-5` | Model to test |

### Server Configuration (for testing)

| Variable | Description |
|----------|-------------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING) |
| `CONFIG_PATH` | Path to config.yaml |

---

## Test Coverage

### Current Coverage: 82%

| Component | Coverage | Status |
|-----------|----------|--------|
| `anthropic_adapter.py` | 100% | Done |
| `openai_adapter.py` | 100% | Done |
| `chain_router.py` | 100% | Done |
| `conversation_routes.py` | 100% | Done |
| `request_queue.py` | 100% | Done |
| `config_models.py` | 100% | Done |
| `api_key_service.py` | 100% | Done |
| `anthropic_routes.py` | 96% | Near complete |
| `message_converter.py` | 96% | Near complete |
| `circuit_breaker.py` | 98% | Near complete |
| `usage_tracker.py` | 95% | Good |
| `middleware.py` | 87% | Good |
| `openai_routes.py` | 74% | Needs work |

### Generate Coverage Report

```bash
# Terminal output
python -m pytest tests/ --cov=app --cov-report=term-missing

# HTML report
python -m pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

---

## For Coding Agents

### Testing Your Integration

If you're building a coding agent that uses ModelSwitch, here's how to test:

#### 1. OpenAI-Compatible API

```python
import httpx

# Non-streaming
resp = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer sk-gateway-admin"},
    json={
        "model": "glm-5",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
    },
    timeout=60.0,
)
print(resp.json())

# Streaming
with httpx.stream(
    "POST",
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer sk-gateway-admin"},
    json={
        "model": "glm-5",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    },
    timeout=60.0,
) as resp:
    for line in resp.iter_lines():
        if line.startswith("data: "):
            print(line)
```

#### 2. Anthropic-Compatible API

```python
import httpx

resp = httpx.post(
    "http://localhost:8000/v1/messages",
    headers={
        "x-api-key": "sk-gateway-admin",
        "anthropic-version": "2023-06-01",
    },
    json={
        "model": "glm-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
    },
    timeout=60.0,
)
print(resp.json())
```

#### 3. Tool Calling

```python
import httpx

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    }
]

resp = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer sk-gateway-admin"},
    json={
        "model": "glm-5",
        "messages": [{"role": "user", "content": "What's the weather in SF?"}],
        "tools": tools,
        "tool_choice": "auto",
    },
    timeout=60.0,
)
print(resp.json())
```

### Common Test Scenarios

| Scenario | Test File | Test Name |
|----------|-----------|-----------|
| Basic chat | `test_e2e.py` | `test_non_stream_chat` |
| Streaming | `test_e2e.py` | `test_stream_chat` |
| Tool calls | `test_e2e.py` | `test_openai_tools_non_stream` |
| Error handling | `test_e2e.py` | `test_model_not_found` |
| Concurrent requests | `test_e2e.py` | `test_concurrent_requests` |
| Auth failure | `test_e2e.py` | `test_no_auth_rejected` |

---

## Troubleshooting

### Tests Fail with "Server not reachable"

**Cause:** Server not running or wrong URL.

**Solution:**
```bash
# Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or set correct URL
MODELSWITCH_URL=http://your-server:8000 pytest tests/test_e2e.py -v
```

### Tests Fail with "Model not found"

**Cause:** Model name in test doesn't match server config.

**Solution:**
```bash
# Check available models
curl http://localhost:8000/v1/models -H "Authorization: Bearer sk-gateway-admin"

# Use correct model name
MODEL_NAME=your-model-name pytest tests/test_e2e.py -v
```

### Tests Fail with "401 Unauthorized"

**Cause:** Invalid API key.

**Solution:**
```bash
# Check config.yaml for valid API keys
# Or use correct key
MODELSWITCH_ADMIN_KEY=your-api-key pytest tests/test_e2e.py -v
```

### Rate Limited During Tests

**Cause:** Too many requests in short time.

**Solution:**
- Increase rate limit in config.yaml
- Run tests with delays
- Use different API keys for concurrent tests

### Import Errors in Tests

**Cause:** Missing dependencies or wrong Python path.

**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run from project root
cd modelswitch
python -m pytest tests/ -v
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ --cov=app --cov-report=xml
      - uses: codecov/codecov-action@v4

  e2e-tests:
    runs-on: ubuntu-latest
    services:
      modelswitch:
        image: modelswitch:latest
        ports:
          - 8000:8000
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements.txt
      - run: MODELSWITCH_E2E=1 pytest tests/test_e2e.py -v
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-push

# Run quick smoke test
python scripts/smoketest.py --quick
if [ $? -ne 0 ]; then
    echo "Smoke tests failed"
    exit 1
fi
```

---

## Test Development Guide

### Adding New Tests

1. **Unit Tests:** Add to appropriate `test_*.py` file
2. **API Tests:** Add to `test_api_routes.py`
3. **E2E Tests:** Add to `test_e2e.py`
4. **Smoke Tests:** Add to both `smoketest.py` and `smoketest.sh`

### Test Naming Convention

```python
# Test class: Test<Feature>
class TestOpenAIChatCompletions:

    # Test method: test_<scenario>
    def test_non_stream_success(self):
        pass

    def test_stream_with_tool_calls(self):
        pass
```

### Using Fixtures

```python
# Available fixtures in conftest.py:
# - client: HTTP client with full app
# - client_with_model_restriction: Client with restricted API key
# - client_rate_limited: Client with rate limiting
# - client_expired_key: Client with expired API key
# - sample_config: Sample configuration
# - usage_db: In-memory usage tracker
```

### Mocking Adapters

```python
from unittest.mock import AsyncMock, patch
from app.adapters.litellm_adapter import AdapterResponse

async def test_example(client, sample_config):
    with patch.object(
        client._transport.app.state.chain_router._adapters.get("provider-a"),
        "chat_completion",
        new_callable=AsyncMock,
        return_value=AdapterResponse(
            status_code=200,
            success=True,
            body={"choices": [{"message": {"content": "Hello"}}]},
            adapter_name="provider-a",
        ),
    ):
        resp = await client.post("/v1/chat/completions", ...)
        assert resp.status_code == 200
```

---

## Summary

| Test Type | Purpose | Run Command |
|-----------|---------|-------------|
| Unit | Component isolation | `pytest tests/ -v` |
| API Integration | HTTP layer | `pytest tests/test_api_routes.py -v` |
| E2E | Full stack | `MODELSWITCH_E2E=1 pytest tests/test_e2e.py -v` |
| Smoke | Quick validation | `python scripts/smoketest.py` |

**Need help?** Check the logs in `logs/gateway.log` or enable debug logging with `LOG_LEVEL=DEBUG`.