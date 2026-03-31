"""Unit tests for config_models and config loading."""
import os
import tempfile

import pytest
import yaml

from app.core.config import load_config, save_config
from app.models.config_models import (
    ApiKeyConfig,
    GatewayConfig,
    ModelConfig,
    ModelAdapterRef,
    ProviderConfig,
    _resolve_env_vars,
    resolve_config_env,
)


class TestResolveEnvVars:
    def test_resolves_known_var(self):
        os.environ["TEST_VAR_XYZ"] = "hello"
        assert _resolve_env_vars("${TEST_VAR_XYZ}") == "hello"
        del os.environ["TEST_VAR_XYZ"]

    def test_keeps_unresolved_var(self):
        result = _resolve_env_vars("${NONEXISTENT_VAR_12345}")
        assert result == "${NONEXISTENT_VAR_12345}"

    def test_non_string_passthrough(self):
        assert _resolve_env_vars(42) == 42
        assert _resolve_env_vars(None) is None
        assert _resolve_env_vars(["list"]) == ["list"]

    def test_mixed_string(self):
        os.environ["MY_HOST"] = "api.example.com"
        result = _resolve_env_vars("https://${MY_HOST}/v1")
        assert result == "https://api.example.com/v1"
        del os.environ["MY_HOST"]

    def test_empty_string(self):
        assert _resolve_env_vars("") == ""

    def test_no_env_vars(self):
        assert _resolve_env_vars("plain string") == "plain string"


class TestResolveConfigEnv:
    def test_resolves_provider_api_key(self):
        os.environ["TEST_RCE_KEY"] = "resolved-key-123"
        cfg = GatewayConfig(
            providers=[ProviderConfig(name="p", provider="openai", base_url="https://x", api_key="${TEST_RCE_KEY}")],
            models={}, api_keys=[],
        )
        result = resolve_config_env(cfg)
        assert result.providers[0].api_key == "resolved-key-123"
        del os.environ["TEST_RCE_KEY"]

    def test_resolves_api_key_config(self):
        os.environ["TEST_RCE_AK"] = "sk-resolved"
        cfg = GatewayConfig(
            providers=[],
            models={},
            api_keys=[ApiKeyConfig(key="${TEST_RCE_AK}", name="test")],
        )
        result = resolve_config_env(cfg)
        assert result.api_keys[0].key == "sk-resolved"
        del os.environ["TEST_RCE_AK"]


class TestProviderConfig:
    def test_defaults(self):
        p = ProviderConfig(name="p", base_url="https://x", api_key="k")
        assert p.provider == "openai"
        assert p.enabled is True
        assert p.custom_headers == {}

    def test_custom_headers(self):
        p = ProviderConfig(name="p", base_url="https://x", api_key="k", custom_headers={"X-Custom": "v"})
        assert p.custom_headers == {"X-Custom": "v"}


class TestModelConfig:
    def test_chain_mode(self):
        m = ModelConfig(mode="chain", adapters=[
            ModelAdapterRef(adapter="a", model_name="m1", priority=1),
            ModelAdapterRef(adapter="b", model_name="m2", priority=2),
        ])
        assert m.mode == "chain"
        assert len(m.adapters) == 2
        assert m.adapters[0].priority == 1

    def test_adapter_mode(self):
        m = ModelConfig(mode="adapter", adapters=[
            ModelAdapterRef(adapter="a", model_name="m1", priority=1),
        ])
        assert m.mode == "adapter"
        assert len(m.adapters) == 1

    def test_defaults(self):
        m = ModelConfig()
        assert m.mode == "chain"
        assert m.adapters == []
        assert m.description == ""


class TestApiKeyConfig:
    def test_defaults(self):
        k = ApiKeyConfig(key="sk-test", name="test")
        assert k.enabled is True
        assert k.rate_limit == 60
        assert k.daily_limit == 0
        assert k.allowed_models == []
        assert k.expires_at is None

    def test_custom_values(self):
        k = ApiKeyConfig(
            key="sk-test", name="test", rate_limit=10,
            daily_limit=100, allowed_models=["glm5"],
            expires_at="2026-12-31T23:59:59",
        )
        assert k.rate_limit == 10
        assert k.daily_limit == 100
        assert k.allowed_models == ["glm5"]


class TestGatewayConfig:
    def test_empty_config(self):
        cfg = GatewayConfig()
        assert cfg.providers == []
        assert cfg.models == {}
        assert cfg.api_keys == []
        assert cfg.gateway.host == "0.0.0.0"
        assert cfg.gateway.port == 8000


class TestLoadConfig:
    def test_loads_yaml_file(self):
        data = {
            "gateway": {"host": "0.0.0.0", "port": 9000},
            "providers": [{"name": "p1", "provider": "openai", "base_url": "https://x", "api_key": "k"}],
            "models": {},
            "api_keys": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg.gateway.port == 9000
            assert len(cfg.providers) == 1
        finally:
            os.unlink(path)

    def test_missing_file_returns_default(self):
        cfg = load_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg, GatewayConfig)
        assert cfg.providers == []

    def test_empty_yaml_returns_default(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        try:
            cfg = load_config(path)
            assert isinstance(cfg, GatewayConfig)
        finally:
            os.unlink(path)


class TestSaveConfig:
    def test_saves_and_roundtrips(self):
        cfg = GatewayConfig(
            providers=[ProviderConfig(name="p1", provider="openai", base_url="https://x", api_key="k")],
            models={"m1": ModelConfig(mode="adapter", adapters=[ModelAdapterRef(adapter="p1", model_name="up")])},
            api_keys=[ApiKeyConfig(key="sk-test", name="test")],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            save_config(cfg, path)
            loaded = load_config(path)
            assert len(loaded.providers) == 1
            assert "m1" in loaded.models
            assert loaded.providers[0].name == "p1"
        finally:
            os.unlink(path)
