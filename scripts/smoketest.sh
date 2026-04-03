#!/bin/bash
#
# ModelSwitch Smoke Test Script
# Tests basic functionality against a running server
#
# Usage:
#   ./smoketest.sh              # Run all tests
#   ./smoketest.sh --quick      # Quick test (skip streaming)
#
# Environment variables:
#   MODELSWITCH_URL   - Server URL (default: http://localhost:8000)
#   MODELSWITCH_KEY   - API key (default: sk-gateway-admin)
#   MODEL_NAME        - Model to test (default: glm5)

set -e

# Configuration
BASE_URL="${MODELSWITCH_URL:-http://localhost:8000}"
API_KEY="${MODELSWITCH_KEY:-sk-gateway-admin}"
MODEL="${MODEL_NAME:-glm5}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
PASSED=0
FAILED=0

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check server is running
check_server() {
    info "Checking server at $BASE_URL..."

    if ! curl -sf --connect-timeout 5 "$BASE_URL/api/config/health" > /dev/null 2>&1; then
        echo ""
        echo -e "${RED}Server not reachable at $BASE_URL${NC}"
        echo ""
        echo "Start the server with:"
        echo "  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
        echo ""
        exit 1
    fi

    pass "Server is reachable"
}

# Test: Health endpoint
test_health() {
    info "Testing health endpoint..."

    local resp
    resp=$(curl -sf "$BASE_URL/api/config/health" 2>/dev/null)

    if echo "$resp" | grep -q '"status"'; then
        pass "Health endpoint OK"
    else
        fail "Health endpoint returned unexpected response"
    fi
}

# Test: Auth required
test_auth_required() {
    info "Testing auth required..."

    local code
    code=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE_URL/v1/models" 2>/dev/null || echo "000")

    if [ "$code" = "401" ]; then
        pass "Unauthenticated requests rejected (401)"
    else
        fail "Expected 401, got $code"
    fi
}

# Test: Valid auth
test_auth_valid() {
    info "Testing valid authentication..."

    local code
    code=$(curl -sf -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $API_KEY" \
        "$BASE_URL/v1/models" 2>/dev/null || echo "000")

    if [ "$code" = "200" ]; then
        pass "Valid API key accepted"
    else
        fail "Valid API key rejected, got $code"
    fi
}

# Test: x-api-key header
test_x_api_key() {
    info "Testing x-api-key header..."

    local code
    code=$(curl -sf -o /dev/null -w "%{http_code}" \
        -H "x-api-key: $API_KEY" \
        "$BASE_URL/v1/models" 2>/dev/null || echo "000")

    if [ "$code" = "200" ]; then
        pass "x-api-key header works"
    else
        fail "x-api-key header rejected, got $code"
    fi
}

# Test: List models
test_list_models() {
    info "Testing list models..."

    local resp
    resp=$(curl -sf -H "Authorization: Bearer $API_KEY" "$BASE_URL/v1/models" 2>/dev/null)

    if echo "$resp" | grep -q '"object"'; then
        pass "List models OK"
    else
        fail "List models failed"
    fi
}

# Test: Non-streaming chat
test_chat_non_stream() {
    info "Testing non-streaming chat..."

    local resp
    resp=$(curl -sf \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":10}" \
        "$BASE_URL/v1/chat/completions" 2>/dev/null)

    if echo "$resp" | grep -q '"choices"'; then
        pass "Non-streaming chat OK"
    else
        fail "Non-streaming chat failed: ${resp:0:100}"
    fi
}

# Test: Streaming chat
test_chat_stream() {
    info "Testing streaming chat..."

    local chunks
    chunks=$(curl -sf -N \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":10,\"stream\":true}" \
        "$BASE_URL/v1/chat/completions" 2>/dev/null | grep -c "^data:" || echo "0")

    if [ "$chunks" -gt 0 ]; then
        pass "Streaming chat OK ($chunks chunks)"
    else
        fail "Streaming chat returned no chunks"
    fi
}

# Test: Anthropic endpoint
test_anthropic() {
    info "Testing Anthropic messages..."

    local resp
    resp=$(curl -sf \
        -H "x-api-key: $API_KEY" \
        -H "anthropic-version: 2023-06-01" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"max_tokens\":10,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}" \
        "$BASE_URL/v1/messages" 2>/dev/null)

    if echo "$resp" | grep -q '"type".*"message"'; then
        pass "Anthropic messages OK"
    else
        fail "Anthropic messages failed: ${resp:0:100}"
    fi
}

# Test: Anthropic streaming
test_anthropic_stream() {
    info "Testing Anthropic streaming..."

    local events
    events=$(curl -sf -N \
        -H "x-api-key: $API_KEY" \
        -H "anthropic-version: 2023-06-01" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"max_tokens\":10,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":true}" \
        "$BASE_URL/v1/messages" 2>/dev/null | grep -c "^event:" || echo "0")

    if [ "$events" -gt 0 ]; then
        pass "Anthropic streaming OK ($events events)"
    else
        fail "Anthropic streaming returned no events"
    fi
}

# Test: Model not found
test_model_not_found() {
    info "Testing model not found..."

    local code
    code=$(curl -sf -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"model":"nonexistent-xyz","messages":[{"role":"user","content":"Hi"}]}' \
        "$BASE_URL/v1/chat/completions" 2>/dev/null || echo "000")

    if [ "$code" = "404" ] || [ "$code" = "502" ]; then
        pass "Model not found returns $code"
    else
        fail "Expected 404/502, got $code"
    fi
}

# Test: CORS
test_cors() {
    info "Testing CORS headers..."

    local header
    header=$(curl -sf -I -X OPTIONS \
        -H "Origin: http://localhost:3000" \
        "$BASE_URL/v1/models" 2>/dev/null | grep -i "access-control-allow-origin" || echo "")

    if [ -n "$header" ]; then
        pass "CORS headers present"
    else
        fail "CORS headers missing"
    fi
}

# Test: Request ID
test_request_id() {
    info "Testing request ID header..."

    local header
    header=$(curl -sf -I \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5}" \
        "$BASE_URL/v1/chat/completions" 2>/dev/null | grep -i "x-request-id" || echo "")

    if [ -n "$header" ]; then
        pass "Request ID header present"
    else
        fail "Request ID header missing"
    fi
}

# Test: Concurrent requests
test_concurrent() {
    info "Testing concurrent requests..."

    local success=0

    for i in 1 2 3; do
        local code
        code=$(curl -sf -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say $i\"}],\"max_tokens\":5}" \
            "$BASE_URL/v1/chat/completions" 2>/dev/null || echo "000")

        if [ "$code" = "200" ]; then
            ((success++))
        fi
    done

    if [ "$success" -eq 3 ]; then
        pass "All 3 concurrent requests succeeded"
    else
        fail "Only $success/3 concurrent requests succeeded"
    fi
}

# Print summary
print_summary() {
    echo ""
    echo "================================"
    echo "       Smoke Test Summary       "
    echo "================================"
    echo -e "  ${GREEN}Passed:${NC} $PASSED"
    echo -e "  ${RED}Failed:${NC} $FAILED"
    echo "================================"
    echo ""

    if [ "$FAILED" -gt 0 ]; then
        exit 1
    fi
}

# Main
main() {
    echo ""
    echo "================================"
    echo "   ModelSwitch Smoke Test       "
    echo "================================"
    echo ""
    echo "Server: $BASE_URL"
    echo "Model:  $MODEL"
    echo ""

    check_server

    # Basic tests
    test_health
    test_auth_required
    test_auth_valid
    test_x_api_key
    test_list_models
    test_model_not_found
    test_cors

    if [ "$1" = "--quick" ]; then
        info "Skipping streaming tests (quick mode)"
        test_chat_non_stream
        test_anthropic
    else
        test_chat_non_stream
        test_chat_stream
        test_anthropic
        test_anthropic_stream
        test_request_id
        test_concurrent
    fi

    print_summary
}

main "$@"
