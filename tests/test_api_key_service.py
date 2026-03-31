"""Unit tests for ApiKeyService."""
import pytest

from app.models.config_models import ApiKeyConfig
from app.services.api_key_service import ApiKeyService


def _make_keys():
    return [
        ApiKeyConfig(key="sk-admin", name="admin", enabled=True, rate_limit=60,
                     daily_limit=0, allowed_models=[], created_at="2026-01-01"),
        ApiKeyConfig(key="sk-user1", name="user1", enabled=True, rate_limit=10,
                     daily_limit=100, allowed_models=["glm5"],
                     created_at="2026-01-01"),
        ApiKeyConfig(key="sk-disabled", name="disabled", enabled=False,
                     created_at="2026-01-01"),
        ApiKeyConfig(key="sk-expired", name="expired", enabled=True,
                     expires_at="2020-01-01T00:00:00", created_at="2020-01-01"),
    ]


class TestGetAll:
    def test_returns_all_keys(self):
        svc = ApiKeyService(_make_keys())
        assert len(svc.get_all()) == 4

    def test_empty_keys(self):
        svc = ApiKeyService([])
        assert svc.get_all() == []


class TestGet:
    def test_existing_key(self):
        svc = ApiKeyService(_make_keys())
        k = svc.get("sk-admin")
        assert k is not None
        assert k.name == "admin"

    def test_nonexistent_key(self):
        svc = ApiKeyService(_make_keys())
        assert svc.get("sk-nope") is None


class TestValidate:
    def test_valid_key(self):
        svc = ApiKeyService(_make_keys())
        k = svc.validate("sk-admin")
        assert k is not None
        assert k.name == "admin"

    def test_missing_key(self):
        svc = ApiKeyService(_make_keys())
        assert svc.validate("sk-nope") is None

    def test_disabled_key(self):
        svc = ApiKeyService(_make_keys())
        assert svc.validate("sk-disabled") is None

    def test_expired_key(self):
        svc = ApiKeyService(_make_keys())
        assert svc.validate("sk-expired") is None

    def test_invalid_expires_at_format_still_valid(self):
        """Key with unparseable expires_at should still validate (ValueError caught)."""
        keys = [ApiKeyConfig(key="sk-bad-date", name="bad-date", enabled=True,
                             expires_at="not-a-date", created_at="2026-01-01")]
        svc = ApiKeyService(keys)
        # Should not crash; ValueError is caught, key stays valid
        k = svc.validate("sk-bad-date")
        assert k is not None


class TestCreate:
    def test_creates_key(self):
        svc = ApiKeyService([])
        k = svc.create(name="new", description="desc", rate_limit=30, daily_limit=0, allowed_models=[])
        assert k.name == "new"
        assert k.key.startswith("sk-")
        assert k.enabled is True
        assert k.rate_limit == 30

    def test_default_name(self):
        svc = ApiKeyService([])
        k = svc.create(name="", description="", rate_limit=60, daily_limit=0, allowed_models=[])
        assert k.name == "unnamed"

    def test_key_is_unique(self):
        svc = ApiKeyService([])
        k1 = svc.create(name="a", description="", rate_limit=60, daily_limit=0, allowed_models=[])
        k2 = svc.create(name="b", description="", rate_limit=60, daily_limit=0, allowed_models=[])
        assert k1.key != k2.key

    def test_added_to_list(self):
        svc = ApiKeyService([])
        svc.create(name="new", description="", rate_limit=60, daily_limit=0, allowed_models=[])
        assert len(svc.get_all()) == 1


class TestUpdate:
    def test_update_fields(self):
        svc = ApiKeyService(_make_keys())
        k = svc.update("sk-admin", rate_limit=10, daily_limit=500)
        assert k is not None
        assert k.rate_limit == 10
        assert k.daily_limit == 500

    def test_update_none_ignored(self):
        svc = ApiKeyService(_make_keys())
        original_limit = svc.get("sk-admin").rate_limit
        k = svc.update("sk-admin", rate_limit=None)
        assert k.rate_limit == original_limit

    def test_update_nonexistent(self):
        svc = ApiKeyService(_make_keys())
        assert svc.update("sk-nope", rate_limit=10) is None

    def test_update_allowed_models(self):
        svc = ApiKeyService(_make_keys())
        k = svc.update("sk-admin", allowed_models=["glm5", "gpt4o"])
        assert k.allowed_models == ["glm5", "gpt4o"]


class TestDelete:
    def test_delete_existing(self):
        svc = ApiKeyService(_make_keys())
        assert svc.delete("sk-admin") is True
        assert svc.get("sk-admin") is None
        assert len(svc.get_all()) == 3

    def test_delete_nonexistent(self):
        svc = ApiKeyService(_make_keys())
        assert svc.delete("sk-nope") is False


class TestToggle:
    def test_toggle_enabled_to_disabled(self):
        svc = ApiKeyService(_make_keys())
        k = svc.toggle("sk-admin")
        assert k.enabled is False

    def test_toggle_disabled_to_enabled(self):
        svc = ApiKeyService(_make_keys())
        k = svc.toggle("sk-disabled")
        assert k.enabled is True

    def test_toggle_nonexistent(self):
        svc = ApiKeyService(_make_keys())
        assert svc.toggle("sk-nope") is None


class TestMaskKey:
    def test_long_key(self):
        assert ApiKeyService.mask_key("sk-1234567890abcdef") == "sk-1234***cdef"

    def test_short_key(self):
        result = ApiKeyService.mask_key("sk-ab")
        assert "***" in result


class TestReload:
    def test_reloads_keys(self):
        svc = ApiKeyService(_make_keys())
        assert len(svc.get_all()) == 4
        new_keys = [ApiKeyConfig(key="sk-new", name="new", created_at="2026-01-01")]
        svc.reload(new_keys)
        assert len(svc.get_all()) == 1
        assert svc.get("sk-new") is not None
        assert svc.get("sk-admin") is None


class TestToList:
    def test_returns_dicts(self):
        svc = ApiKeyService(_make_keys())
        result = svc.to_list()
        assert isinstance(result, list)
        assert len(result) == 4
        # Should have key_raw field
        assert any(k.get("key_raw") for k in result)
        # Keys should be masked in 'key' field
        for k in result:
            assert "***" in k["key"]
