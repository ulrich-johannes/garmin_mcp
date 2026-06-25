"""Starlette ASGI middleware that enforces bearer-token authentication.

The middleware sits in front of the FastMCP Starlette app.  For every HTTP and
WebSocket request it:
  1. Extracts the Bearer token from the Authorization header.
  2. Calls auth_provider.resolve(token).  None → 401, UserContext → proceed.
  3. On the first request from a new user (user_id changed), calls on_new_client
     so the caller can configure module-level Garmin client globals.
  4. Stores the resolved UserContext in scope["garmin_user"] for downstream use.

The /healthz path is exempt from auth (k8s liveness probes, uptime checks).

Phase 2 note: when moving to per-request client injection via contextvars, the
on_new_client callback is replaced by setting a ContextVar here instead.  The
AuthProvider interface and this middleware class stay unchanged.
"""

import sys
from typing import Callable

from garmin_mcp.auth.base import AuthProvider, UserContext

_PUBLIC_PATHS = {
    "/healthz",
    "/authorize",
    "/oauth/token",
    "/register",
    "/.well-known/oauth-authorization-server",
}


def _extract_bearer(scope: dict) -> str | None:
    """Extract the bearer token from Authorization header or ?token= query param.

    Priority: Authorization header > ?token= query parameter.
    The query-param fallback exists because the claude.ai connector UI has no
    bearer-token field — it only supports OAuth 2.0.  Embedding the token in
    the URL is a pragmatic workaround for personal / single-user use.
    """
    headers = dict(scope.get("headers", []))
    raw = headers.get(b"authorization", b"").decode("utf-8", errors="replace")
    if raw.lower().startswith("bearer "):
        return raw[7:].strip() or None

    # Fall back to ?token= query parameter (claude.ai connector UI)
    query = scope.get("query_string", b"").decode("utf-8", errors="replace")
    for part in query.split("&"):
        if part.startswith("token="):
            return part[6:] or None

    return None


async def _send_401(send: Callable) -> None:
    body = b'{"error":"Unauthorized","detail":"Valid Garmin base64 token required as Bearer token"}'
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [
            [b"content-type", b"application/json"],
            [b"content-length", str(len(body)).encode()],
            [b"www-authenticate", b'Bearer realm="garmin-mcp"'],
        ],
    })
    await send({"type": "http.response.body", "body": body})


class GarminAuthMiddleware:
    """ASGI middleware wrapping any Starlette app with pluggable bearer-token auth.

    Args:
        app:            The inner ASGI app (FastMCP Starlette instance).
        auth_provider:  Validates tokens and resolves them to a UserContext.
        on_new_client:  Optional callback invoked with the raw Garmin client
                        whenever the resolved user changes.  Use this to
                        configure module-level garmin_client globals (Phase 1)
                        or set a ContextVar (Phase 2).
    """

    def __init__(
        self,
        app,
        auth_provider: AuthProvider,
        on_new_client: Callable | None = None,
    ) -> None:
        self._app = app
        self._auth_provider = auth_provider
        self._on_new_client = on_new_client
        self._last_user_id: str | None = None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            # lifespan and other ASGI events pass through unconditionally
            await self._app(scope, receive, send)
            return

        # Public paths: health probe + OAuth endpoints
        if scope.get("path") in _PUBLIC_PATHS:
            await self._app(scope, receive, send)
            return

        token = _extract_bearer(scope)
        if not token:
            await _send_401(send)
            return

        ctx: UserContext | None = self._auth_provider.resolve(token)
        if ctx is None:
            await _send_401(send)
            return

        # Configure module globals when the resolved user changes.
        # For Phase 1 (single user) this fires once on the first request and
        # is a no-op on every subsequent request.
        if ctx.user_id != self._last_user_id:
            if self._on_new_client is not None:
                self._on_new_client(ctx.garmin_client)
            self._last_user_id = ctx.user_id
            print(
                f"[garmin-mcp] Authenticated user {ctx.user_id[:8]}…",
                file=sys.stderr,
            )

        scope["garmin_user"] = ctx
        await self._app(scope, receive, send)
