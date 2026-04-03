#!/usr/bin/env python3
"""
ModelSwitch Smoke Test Script
Tests basic functionality against a running server.

Usage:
    python smoketest.py              # Run all tests
    python smoketest.py --quick      # Quick test (skip streaming)
    python smoketest.py --help       # Show help

Environment variables:
    MODELSWITCH_URL   - Server URL (default: http://localhost:8000)
    MODELSWITCH_KEY   - API key (default: sk-gateway-admin)
    MODEL_NAME        - Model to test (default: glm5)
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import httpx

# Configuration
BASE_URL = os.environ.get("MODELSWITCH_URL", "http://localhost:8000")
API_KEY = os.environ.get("MODELSWITCH_KEY", "sk-gateway-admin")
MODEL = os.environ.get("MODEL_NAME", "glm5")
TIMEOUT = 60.0


# Colors for output
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    RESET = "\033[0m"


# Test counters
passed = 0
failed = 0


def colorize(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"


def log_pass(message: str):
    """Log a passed test."""
    global passed
    print(colorize(f"[PASS] {message}", Colors.GREEN))
    passed += 1


def log_fail(message: str):
    """Log a failed test."""
    global failed
    print(colorize(f"[FAIL] {message}", Colors.RED))
    failed += 1


def log_info(message: str):
    """Log info message."""
    print(colorize(f"[INFO] {message}", Colors.BLUE))


def log_skip(message: str):
    """Log skipped test."""
    print(colorize(f"[SKIP] {message}", Colors.YELLOW))


def check_server() -> bool:
    """Check if server is running."""
    log_info(f"Checking server at {BASE_URL}...")
    try:
        resp = httpx.get(f"{BASE_URL}/api/config/health", timeout=5.0)
        if resp.status_code == 200:
            log_pass("Server is reachable")
            return True
    except Exception as e:
        print()
        print(colorize(f"Server not reachable at {BASE_URL}", Colors.RED))
        print()
        print("Start the server with:")
        print("  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        print()
        print("Or set MODELSWITCH_URL environment variable:")
        print("  MODELSWITCH_URL=http://your-server:8000 python smoketest.py")
        print()
        sys.exit(1)
    return False


def test_health():
    """Test health endpoint."""
    log_info("Testing health endpoint...")
    try:
        resp = httpx.get(f"{BASE_URL}/api/config/health", timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "healthy":
                log_pass("Health endpoint returns healthy status")
            else:
                log_fail(f"Health endpoint returned: {data}")
        else:
            log_fail(f"Health endpoint returned status {resp.status_code}")
    except Exception as e:
        log_fail(f"Health endpoint error: {e}")


def test_auth_required():
    """Test that auth is required."""
    log_info("Testing auth required...")
    try:
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=10.0)
        if resp.status_code == 401:
            log_pass("Unauthenticated requests rejected (401)")
        else:
            log_fail(f"Expected 401, got {resp.status_code}")
    except Exception as e:
        log_fail(f"Auth test error: {e}")


def test_auth_valid():
    """Test valid authentication."""
    log_info("Testing valid authentication...")
    try:
        resp = httpx.get(
            f"{BASE_URL}/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            log_pass("Valid API key accepted")
        else:
            log_fail(f"Valid API key rejected, got {resp.status_code}")
    except Exception as e:
        log_fail(f"Auth valid test error: {e}")


def test_x_api_key():
    """Test x-api-key header authentication."""
    log_info("Testing x-api-key header...")
    try:
        resp = httpx.get(
            f"{BASE_URL}/v1/models",
            headers={"x-api-key": API_KEY},
            timeout=10.0,
        )
        if resp.status_code == 200:
            log_pass("x-api-key header works")
        else:
            log_fail(f"x-api-key header rejected, got {resp.status_code}")
    except Exception as e:
        log_fail(f"x-api-key test error: {e}")


def test_list_models():
    """Test list models endpoint."""
    log_info("Testing list models...")
    try:
        resp = httpx.get(
            f"{BASE_URL}/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("object") == "list" and "data" in data:
                model_count = len(data["data"])
                log_pass(f"List models returned {model_count} models")
            else:
                log_fail("List models returned unexpected format")
        else:
            log_fail(f"List models returned status {resp.status_code}")
    except Exception as e:
        log_fail(f"List models error: {e}")


def test_chat_non_stream():
    """Test non-streaming chat completion."""
    log_info("Testing non-streaming chat...")
    try:
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "Say 'hello world' and nothing else."}
                ],
                "max_tokens": 20,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                log_pass(f"Non-streaming chat OK: {content[:50]}...")
            else:
                log_fail("Non-streaming chat missing choices")
        else:
            log_fail(
                f"Non-streaming chat returned status {resp.status_code}: {resp.text[:100]}"
            )
    except Exception as e:
        log_fail(f"Non-streaming chat error: {e}")


def test_chat_stream():
    """Test streaming chat completion."""
    log_info("Testing streaming chat...")
    try:
        with httpx.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Count from 1 to 3."}],
                "max_tokens": 20,
                "stream": True,
            },
            timeout=TIMEOUT,
        ) as resp:
            if resp.status_code == 200:
                chunks = 0
                has_done = False
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        if line == "data: [DONE]":
                            has_done = True
                        else:
                            chunks += 1
                if chunks > 0 and has_done:
                    log_pass(f"Streaming chat OK ({chunks} chunks)")
                else:
                    log_fail(f"Streaming incomplete: {chunks} chunks, done={has_done}")
            else:
                log_fail(f"Streaming chat returned status {resp.status_code}")
    except Exception as e:
        log_fail(f"Streaming chat error: {e}")


def test_anthropic():
    """Test Anthropic messages endpoint."""
    log_info("Testing Anthropic messages...")
    try:
        resp = httpx.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 20,
                "messages": [
                    {"role": "user", "content": "Say 'hello' and nothing else."}
                ],
            },
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("type") == "message":
                log_pass("Anthropic messages OK")
            else:
                log_fail(f"Anthropic messages unexpected format: {data.get('type')}")
        else:
            log_fail(
                f"Anthropic messages returned status {resp.status_code}: {resp.text[:100]}"
            )
    except Exception as e:
        log_fail(f"Anthropic messages error: {e}")


def test_anthropic_stream():
    """Test Anthropic streaming."""
    log_info("Testing Anthropic streaming...")
    try:
        with httpx.stream(
            "POST",
            f"{BASE_URL}/v1/messages",
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
            timeout=TIMEOUT,
        ) as resp:
            if resp.status_code == 200:
                events = 0
                for line in resp.iter_lines():
                    if line.startswith("event:") or line.startswith("data:"):
                        events += 1
                if events > 0:
                    log_pass(f"Anthropic streaming OK ({events} events)")
                else:
                    log_fail("Anthropic streaming returned no events")
            else:
                log_fail(f"Anthropic streaming returned status {resp.status_code}")
    except Exception as e:
        log_fail(f"Anthropic streaming error: {e}")


def test_model_not_found():
    """Test model not found error."""
    log_info("Testing model not found...")
    try:
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "nonexistent-model-xyz",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=10.0,
        )
        if resp.status_code in [404, 502]:
            log_pass(f"Model not found returns {resp.status_code}")
        else:
            log_fail(f"Expected 404/502, got {resp.status_code}")
    except Exception as e:
        log_fail(f"Model not found test error: {e}")


def test_cors():
    """Test CORS headers."""
    log_info("Testing CORS headers...")
    try:
        resp = httpx.options(
            f"{BASE_URL}/v1/models",
            headers={"Origin": "http://localhost:3000"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            cors_header = resp.headers.get("access-control-allow-origin")
            if cors_header:
                log_pass(f"CORS headers present: {cors_header}")
            else:
                log_fail("CORS headers missing")
        else:
            log_fail(f"CORS preflight returned status {resp.status_code}")
    except Exception as e:
        log_fail(f"CORS test error: {e}")


def test_request_id():
    """Test request ID header."""
    log_info("Testing request ID header...")
    try:
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            },
            timeout=TIMEOUT,
        )
        request_id = resp.headers.get("x-request-id")
        if request_id:
            log_pass(f"Request ID header present: {request_id}")
        else:
            log_fail("Request ID header missing")
    except Exception as e:
        log_fail(f"Request ID test error: {e}")


def test_concurrent():
    """Test concurrent requests."""
    log_info("Testing concurrent requests...")
    import concurrent.futures

    def make_request(i: int) -> int:
        try:
            resp = httpx.post(
                f"{BASE_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": f"Say {i}"}],
                    "max_tokens": 5,
                },
                timeout=TIMEOUT,
            )
            return resp.status_code
        except Exception:
            return 0

    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(make_request, i) for i in range(3)]
        for future in concurrent.futures.as_completed(futures):
            if future.result() == 200:
                success += 1

    if success == 3:
        log_pass("All 3 concurrent requests succeeded")
    else:
        log_fail(f"Only {success}/3 concurrent requests succeeded")


def test_tool_calls():
    """Test tool calling functionality."""
    log_info("Testing tool calls...")
    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "What's the weather in San Francisco?"}
                ],
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 100,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            if "choices" in data:
                log_pass("Tool calls endpoint accessible")
            else:
                log_fail("Tool calls response missing choices")
        else:
            log_fail(f"Tool calls returned status {resp.status_code}")
    except Exception as e:
        log_fail(f"Tool calls test error: {e}")


def print_summary():
    """Print test summary."""
    print()
    print("=" * 40)
    print("         Smoke Test Summary")
    print("=" * 40)
    print(f"  {colorize('Passed:', Colors.GREEN)} {passed}")
    print(f"  {colorize('Failed:', Colors.RED)} {failed}")
    print("=" * 40)
    print()

    if failed > 0:
        sys.exit(1)


def main():
    """Main entry point."""
    global BASE_URL, API_KEY, MODEL

    parser = argparse.ArgumentParser(description="ModelSwitch Smoke Test")
    parser.add_argument(
        "--quick", action="store_true", help="Quick test (skip streaming)"
    )
    parser.add_argument("--url", default=BASE_URL, help="Server URL")
    parser.add_argument("--key", default=API_KEY, help="API key")
    parser.add_argument("--model", default=MODEL, help="Model to test")
    args = parser.parse_args()

    BASE_URL = args.url
    API_KEY = args.key
    MODEL = args.model

    print()
    print("=" * 40)
    print("      ModelSwitch Smoke Test")
    print("=" * 40)
    print()
    print(f"Server: {BASE_URL}")
    print(f"Model:  {MODEL}")
    print()

    check_server()

    # Basic tests
    test_health()
    test_auth_required()
    test_auth_valid()
    test_x_api_key()
    test_list_models()
    test_model_not_found()
    test_cors()

    if args.quick:
        log_skip("Streaming tests (quick mode)")
        test_chat_non_stream()
        test_anthropic()
    else:
        test_chat_non_stream()
        test_chat_stream()
        test_anthropic()
        test_anthropic_stream()
        test_request_id()
        test_concurrent()
        test_tool_calls()

    print_summary()


if __name__ == "__main__":
    main()
