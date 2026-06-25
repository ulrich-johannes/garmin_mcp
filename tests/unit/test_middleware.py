"""Unit tests for GarminAuthMiddleware."""

import pytest
from unittest.mock import AsyncMock, MagicMock, call

from garmin_mcp.auth.base import AuthProvider, UserContext
from garmin_mcp.middleware import GarminAuthMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scope(
    path: str = "/mcp",
    auth_header: bytes | None = None,
    query_string: bytes = b"",
) -> dict:
    headers = []
    if auth_header is not None:
        headers.append((b"authorization", auth_header))
    return {"type": "http", "path": path, "headers": headers, "query_string": query_string}


class _StaticAuthProvider(AuthProvider):
    """Test double: returns a fixed UserContext for a specific valid token."""

    def __init__(self, valid_token: str, user_id: str = "testuser01234567"):
        self._valid = valid_token
        self._ctx = UserContext(user_id=user_id, garmin_client=MagicMock())

    def resolve(self, token: str) -> UserContext | None:
        return self._ctx if token == self._valid else None


def _middleware(
    valid_token: str = "good-token",
    on_new_client=None,
) -> GarminAuthMiddleware:
    inner = AsyncMock()  # inner ASGI app
    provider = _StaticAuthProvider(valid_token)
    mw = GarminAuthMiddleware(inner, provider, on_new_client=on_new_client)
    mw._inner = inner  # expose for assertions
    return mw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGarminAuthMiddleware:

    @pytest.mark.asyncio
    async def test_missing_auth_header_returns_401(self):
        mw = _middleware()
        scope = _make_scope()
        send = AsyncMock()

        await mw(scope, AsyncMock(), send)

        mw._inner.assert_not_called()
        first_call_args = send.call_args_list[0][0][0]
        assert first_call_args["status"] == 401

    @pytest.mark.asyncio
    async def test_wrong_token_returns_401(self):
        mw = _middleware(valid_token="secret")
        scope = _make_scope(auth_header=b"Bearer wrong-token")
        send = AsyncMock()

        await mw(scope, AsyncMock(), send)

        mw._inner.assert_not_called()
        first_call_args = send.call_args_list[0][0][0]
        assert first_call_args["status"] == 401

    @pytest.mark.asyncio
    async def test_valid_token_reaches_inner_app(self):
        mw = _middleware(valid_token="secret")
        scope = _make_scope(auth_header=b"Bearer secret")
        receive, send = AsyncMock(), AsyncMock()

        await mw(scope, receive, send)

        mw._inner.assert_awaited_once_with(scope, receive, send)

    @pytest.mark.asyncio
    async def test_valid_token_sets_garmin_user_in_scope(self):
        mw = _middleware(valid_token="secret")
        scope = _make_scope(auth_header=b"Bearer secret")

        await mw(scope, AsyncMock(), AsyncMock())

        assert "garmin_user" in scope
        assert isinstance(scope["garmin_user"], UserContext)

    @pytest.mark.asyncio
    async def test_healthz_skips_auth(self):
        mw = _middleware(valid_token="secret")
        scope = _make_scope(path="/healthz")  # no auth header
        receive, send = AsyncMock(), AsyncMock()

        await mw(scope, receive, send)

        mw._inner.assert_awaited_once_with(scope, receive, send)

    @pytest.mark.asyncio
    async def test_oauth_token_path_skips_auth(self):
        mw = _middleware(valid_token="secret")
        scope = _make_scope(path="/oauth/token")
        receive, send = AsyncMock(), AsyncMock()

        await mw(scope, receive, send)

        mw._inner.assert_awaited_once_with(scope, receive, send)

    @pytest.mark.asyncio
    async def test_oauth_metadata_path_skips_auth(self):
        mw = _middleware(valid_token="secret")
        scope = _make_scope(path="/.well-known/oauth-authorization-server")
        receive, send = AsyncMock(), AsyncMock()

        await mw(scope, receive, send)

        mw._inner.assert_awaited_once_with(scope, receive, send)

    @pytest.mark.asyncio
    async def test_on_new_client_called_on_first_request(self):
        on_new_client = MagicMock()
        mw = _middleware(valid_token="secret", on_new_client=on_new_client)
        scope = _make_scope(auth_header=b"Bearer secret")

        await mw(scope, AsyncMock(), AsyncMock())

        on_new_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_new_client_not_called_again_for_same_user(self):
        on_new_client = MagicMock()
        mw = _middleware(valid_token="secret", on_new_client=on_new_client)
        scope = _make_scope(auth_header=b"Bearer secret")

        await mw(scope, AsyncMock(), AsyncMock())
        await mw(scope, AsyncMock(), AsyncMock())

        on_new_client.assert_called_once()  # not twice

    @pytest.mark.asyncio
    async def test_token_query_param_accepted(self):
        mw = _middleware(valid_token="secret")
        scope = _make_scope(query_string=b"token=secret")
        receive, send = AsyncMock(), AsyncMock()

        await mw(scope, receive, send)

        mw._inner.assert_awaited_once_with(scope, receive, send)

    @pytest.mark.asyncio
    async def test_header_takes_priority_over_query_param(self):
        """Authorization header wins when both are present."""
        mw = _middleware(valid_token="header-token")
        scope = _make_scope(
            auth_header=b"Bearer header-token",
            query_string=b"token=wrong-token",
        )
        receive, send = AsyncMock(), AsyncMock()

        await mw(scope, receive, send)

        mw._inner.assert_awaited_once_with(scope, receive, send)

    @pytest.mark.asyncio
    async def test_lifespan_events_pass_through_without_auth(self):
        mw = _middleware(valid_token="secret")
        scope = {"type": "lifespan"}
        receive, send = AsyncMock(), AsyncMock()

        await mw(scope, receive, send)

        mw._inner.assert_awaited_once_with(scope, receive, send)
