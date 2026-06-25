"""Unit tests for GarminTokenAuthProvider."""

import base64
import json
import functools
from unittest.mock import MagicMock, patch

import pytest

from garmin_mcp.auth.garmin_token import GarminTokenAuthProvider, _cached_garmin_client


def _make_token(payload: dict | None = None) -> str:
    """Build a fake base64 Garmin token from a JSON dict."""
    data = payload if payload is not None else {"access_token": "fake", "token_type": "Bearer"}
    return base64.b64encode(json.dumps(data).encode()).decode()


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Clear the module-level LRU cache between tests so they are independent."""
    _cached_garmin_client.cache_clear()
    yield
    _cached_garmin_client.cache_clear()


class TestGarminTokenAuthProvider:

    def _provider(self, is_cn: bool = False) -> GarminTokenAuthProvider:
        return GarminTokenAuthProvider(is_cn=is_cn)

    # --- happy path -----------------------------------------------------------

    def test_valid_token_returns_user_context(self):
        token = _make_token()
        mock_client = MagicMock()

        with patch("garmin_mcp.auth.garmin_token._init_garmin_from_base64", return_value=mock_client):
            ctx = self._provider().resolve(token)

        assert ctx is not None
        assert ctx.garmin_client is mock_client
        assert len(ctx.user_id) == 16  # sha256 hex prefix

    def test_same_token_returns_same_user_id(self):
        token = _make_token()
        mock_client = MagicMock()

        with patch("garmin_mcp.auth.garmin_token._init_garmin_from_base64", return_value=mock_client):
            ctx1 = self._provider().resolve(token)
            ctx2 = self._provider().resolve(token)

        assert ctx1.user_id == ctx2.user_id

    def test_different_tokens_produce_different_user_ids(self):
        token_a = _make_token({"user": "alice"})
        token_b = _make_token({"user": "bob"})
        mock_client = MagicMock()

        with patch("garmin_mcp.auth.garmin_token._init_garmin_from_base64", return_value=mock_client):
            ctx_a = self._provider().resolve(token_a)
            ctx_b = self._provider().resolve(token_b)

        assert ctx_a.user_id != ctx_b.user_id

    def test_lru_cache_reuses_client(self):
        token = _make_token()
        mock_client = MagicMock()

        with patch("garmin_mcp.auth.garmin_token._init_garmin_from_base64", return_value=mock_client) as mock_init:
            self._provider().resolve(token)
            self._provider().resolve(token)

        # _init_garmin_from_base64 called only once; second resolve is a cache hit
        mock_init.assert_called_once()

    # --- failure paths --------------------------------------------------------

    def test_invalid_base64_returns_none(self):
        ctx = self._provider().resolve("not-valid-base64!!!")
        assert ctx is None

    def test_invalid_json_after_decode_returns_none(self):
        bad_token = base64.b64encode(b"this is not json").decode()
        ctx = self._provider().resolve(bad_token)
        assert ctx is None

    def test_garmin_login_failure_returns_none(self):
        token = _make_token()

        with patch(
            "garmin_mcp.auth.garmin_token._init_garmin_from_base64",
            side_effect=Exception("auth failed"),
        ):
            ctx = self._provider().resolve(token)

        assert ctx is None

    def test_garmin_login_failure_does_not_cache_error(self):
        """A failed login should NOT be cached; next attempt must retry."""
        token = _make_token()
        mock_client = MagicMock()

        with patch(
            "garmin_mcp.auth.garmin_token._init_garmin_from_base64",
            side_effect=[Exception("transient error"), mock_client],
        ) as mock_init:
            ctx1 = self._provider().resolve(token)  # fails
            # lru_cache does not cache exceptions, so the second call retries
            ctx2 = self._provider().resolve(token)  # succeeds

        assert ctx1 is None
        assert ctx2 is not None
        assert mock_init.call_count == 2
