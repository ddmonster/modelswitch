# TODO: Test Coverage Improvements

This document tracks test coverage improvements for ModelSwitch.

## Current Status (82%)

### Done (100% Coverage)

| Component | Coverage | Status |
|-----------|----------|--------|
| `app/adapters/anthropic_adapter.py` | 100% | Done |
| `app/adapters/openai_adapter.py` | 100% | Done |
| `app/core/chain_router.py` | 100% | Done |
| `app/api/conversation_routes.py` | 100% | Done |
| `app/core/request_queue.py` | 100% | Done |
| `app/models/config_models.py` | 100% | Done |
| `app/services/api_key_service.py` | 100% | Done |

### Near Complete (Critical/High Priority)

| Component | Coverage | Missing Lines |
|-----------|----------|---------------|
| `app/api/anthropic_routes.py` | 96% | 79, 178, 182-183, 200 |
| `app/api/openai_routes.py` | 74% | 69, 116-119, 150-157, 171, 173, 179-196, 200-202, 215, 217-218, 220 |

---

## End-to-End & Smoke Tests

### E2E Tests (`tests/test_e2e.py`)

End-to-end tests with real HTTP requests against a running server.

**Usage:**
```bash
# Start server first
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run e2e tests
MODELSWITCH_E2E=1 pytest tests/test_e2e.py -v

# With custom server
MODELSWITCH_E2E=1 MODELSWITCH_URL=http://localhost:8000 pytest tests/test_e2e.py -v
```

**Test Coverage:**
- Health and config endpoints
- Authentication (Bearer, x-api-key, no auth)
- OpenAI chat completions (stream/non-stream)
- Anthropic messages (stream/non-stream)
- Tool calling functionality
- Error handling (invalid JSON, missing params)
- CORS headers
- Request ID headers
- Concurrent request handling
- Streaming behavior (SSE format, cancellation, timeout)

### Smoke Tests

**Python version (`scripts/smoketest.py`):**
```bash
# Run all tests
python scripts/smoketest.py

# Quick test (skip streaming)
python scripts/smoketest.py --quick

# Custom server
python scripts/smoketest.py --url http://localhost:8000 --key sk-gateway-admin --model glm5
```

**Bash version (`scripts/smoketest.sh`):**
```bash
# Run all tests
./scripts/smoketest.sh

# Quick test (skip streaming)
./scripts/smoketest.sh --quick

# Custom server
MODELSWITCH_URL=http://localhost:8000 MODELSWITCH_KEY=sk-gateway-admin ./scripts/smoketest.sh
```

**Smoke Test Coverage:**
- Server health check
- Authentication (401 without auth, 200 with valid auth)
- x-api-key header authentication
- List models endpoint
- Non-streaming chat completion
- Streaming chat completion
- Anthropic messages endpoint
- Anthropic streaming
- Model not found error
- CORS headers
- Request ID header
- Concurrent requests (3 simultaneous)
- Tool calls

---

## Medium Priority Tests

### 1. `app/core/middleware.py` - 87%

Missing lines: 41-42, 50-56, 62-63, 95-96, 126-131, 145-146, 168, 213

**Tests needed:**
- [ ] Request timeout handling
- [ ] Large request body handling
- [ ] WebSocket upgrade handling
- [ ] Error response formatting for different accept headers
- [ ] Rate limit retry-after header calculation
- [ ] CORS preflight with custom headers

### 2. `app/api/config_routes.py` - 85%

Missing lines: 31-46, 117, 163, 167, 209-212, 235, 267, 285-287, 313-322, 325-334, 354-357, 381-383

**Tests needed:**
- [ ] Provider test endpoint with various error scenarios
- [ ] Model test endpoint with streaming responses
- [ ] Config validation errors
- [ ] Duplicate provider/model handling
- [ ] Config file write failures

### 3. `app/core/circuit_breaker.py` - 98%

Missing lines: 41

**Tests needed:**
- [ ] Half-open state with concurrent requests

### 4. `app/services/usage_tracker.py` - 95%

Missing lines: 35, 199-202

**Tests needed:**
- [ ] Database connection error handling
- [ ] Flush with pending records on close

### 5. `app/api/usage_routes.py` - 82%

Missing lines: 25-26, 54, 56, 58

**Tests needed:**
- [ ] Usage detail with no data
- [ ] Date range filtering
- [ ] Invalid group_by parameter handling

### 6. `app/api/api_key_routes.py` - 96%

Missing lines: 68, 101, 122

**Tests needed:**
- [ ] Create key with existing name
- [ ] Update key with invalid data
- [ ] Delete key that's in use

### 7. `app/core/exceptions.py` - 57%

Missing lines: 7-10, 15, 24, 33, 42, 51-56

**Tests needed:**
- [ ] All exception classes instantiation
- [ ] Exception message formatting
- [ ] Exception inheritance chain

### 8. `app/utils/tracking.py` - 89%

Missing lines: 22-25, 40, 43, 108-109

**Tests needed:**
- [ ] Track request with missing usage tracker
- [ ] Track streaming request with no output
- [ ] Log buffer integration

### 9. `app/utils/message_converter.py` - 96%

Missing lines: 164, 168, 250, 273-275

**Tests needed:**
- [ ] Empty tool result content
- [ ] Malformed tool result blocks
- [ ] Stream with malformed chunks

---

## Low Priority Tests

### 1. `app/core/config_watcher.py` - 0%

**Tests needed:**
- [ ] Config file change detection
- [ ] Debounce mechanism (2s delay)
- [ ] Hot reload with invalid config
- [ ] Watcher start/stop lifecycle

### 2. `app/utils/metrics.py` - 0%

**Tests needed:**
- [ ] Prometheus metrics endpoint
- [ ] Request counter increment
- [ ] Latency histogram observation
- [ ] Circuit breaker state gauge
- [ ] Active requests gauge

### 3. `app/main.py` - 0%

**Tests needed:**
- [ ] Application startup
- [ ] Application shutdown
- [ ] Signal handling (SIGTERM, SIGINT)
- [ ] Config watcher initialization
- [ ] Usage tracker flush on shutdown

### 4. `app/utils/logging.py` - 47%

Missing lines: 26-33, 43-81, 99, 120, 129-164

**Tests needed:**
- [ ] Setup logging with custom level
- [ ] JSON formatter output
- [ ] Conversation logger
- [ ] Log file rotation
- [ ] LoggingMiddleware error handling

### 5. `app/adapters/litellm_adapter.py` - 90%

Missing lines: 16

**Tests needed:**
- [ ] LiteLLM adapter initialization
- [ ] Model name prefix handling

### 6. `app/adapters/base.py` - 80%

Missing lines: 51-63

**Tests needed:**
- [ ] BaseAdapter abstract methods
- [ ] Provider config validation

---

## Specific Scenarios for Coding Agents

The following scenarios are important for coding agents like OpenCode, Claude Code, and OpenAI Agents Python:

### Tool Calling (Partially Covered)
- [x] Static tool_calls conversion
- [x] Streaming tool_calls delta
- [ ] Multi-turn tool conversations (user → assistant tool_call → tool_result → assistant response)
- [ ] Tool choice validation (auto, required, none, named function)

### Streaming (Partially Covered)
- [x] Basic streaming response
- [x] Streaming with error mid-stream
- [ ] Streaming timeout handling
- [ ] Streaming with large content

### Error Handling (Partially Covered)
- [x] API timeout errors
- [x] API status errors
- [ ] Rate limit error with retry-after
- [ ] Invalid request error formatting

### Concurrent Requests
- [ ] Multiple simultaneous requests from same API key
- [ ] Request queue overflow
- [ ] Priority queue ordering

### Image/Vision Content
- [ ] Messages with base64 images
- [ ] Messages with image URLs
- [ ] Vision model support

### Response Format
- [ ] JSON mode (`response_format: {"type": "json_object"}`)
- [ ] Structured output validation

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_api_routes.py -v

# Run with HTML coverage report
python -m pytest tests/ --cov=app --cov-report=html
```

---

## Notes

1. **Critical/High priority items** focus on core functionality that coding agents depend on:
   - API adapters (anthropic, openai)
   - Request routing (chain_router)
   - Request queuing
   - Streaming responses

2. **Medium priority items** are important for production reliability:
   - Error handling
   - Metrics and monitoring
   - Usage tracking

3. **Low priority items** are nice-to-have:
   - Config hot reload
   - Startup/shutdown
   - Logging utilities

---

## Last Updated

Date: 2025-01-09
Overall Coverage: 82%
Tests: 400+ passing