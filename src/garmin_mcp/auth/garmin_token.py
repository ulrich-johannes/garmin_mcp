"""GarminTokenAuthProvider — bearer token IS the Garmin OAuth token (base64-encoded).

The caller runs `garmin-mcp-auth` locally once, gets `~/.garminconnect_base64`, and
pastes its contents as the `authorization_token` in their Claude MCP connector config.
No server-side token storage is required; the server is fully stateless.

Resolved clients are cached by token hash (LRU, max 32 entries) so repeated calls within
a Claude session don't re-run the Garmin login for every tool invocation.

Phase 2 upgrade path: replace this provider with one that maps per-user tokens stored in
a database and pass the new provider to GarminAuthMiddleware — nothing else changes.
"""

import base64
import functools
import hashlib
import json
import os
import shutil
import sys
import tempfile

from garmin_mcp.auth.base import AuthProvider, UserContext


def _init_garmin_from_base64(token_b64: str, is_cn: bool) -> object:
    """Decode a base64 Garmin token, write it to a secure temp dir, and log in.

    The temp dir is deleted immediately after login; garminconnect loads token
    state into memory so the files are not needed after the call returns.

    Raises on any failure (bad base64, invalid JSON, expired/revoked token).
    The lru_cache in _cached_garmin_client does NOT cache exceptions, so a bad
    token retries on every request rather than staying permanently rejected.
    """
    from garminconnect import Garmin

    # Decode and sanity-check before touching the filesystem
    try:
        token_json = base64.b64decode(token_b64).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Bearer token is not valid base64: {exc}") from exc

    try:
        json.loads(token_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Decoded token is not valid JSON: {exc}") from exc

    tmpdir = tempfile.mkdtemp(prefix="garmin_mcp_auth_")
    try:
        token_file = os.path.join(tmpdir, "garmin_tokens.json")
        with open(token_file, "w") as fh:
            fh.write(token_json)
        os.chmod(tmpdir, 0o700)
        os.chmod(token_file, 0o600)

        garmin = Garmin(is_cn=is_cn)
        garmin.login(tmpdir)
        return garmin
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@functools.lru_cache(maxsize=32)
def _cached_garmin_client(token_b64: str, is_cn: bool) -> object:
    """LRU-cached Garmin client keyed by (token, region).

    maxsize=32 supports up to 32 distinct users concurrently.  For Phase 1
    (personal use) there is effectively one slot ever used.
    """
    return _init_garmin_from_base64(token_b64, is_cn)


class GarminTokenAuthProvider(AuthProvider):
    """Validates that the bearer token is a working Garmin OAuth token (base64).

    resolve() returns None on any failure: bad encoding, invalid JSON, or Garmin
    authentication error (expired/revoked).  Errors are logged to stderr so the
    server operator can diagnose issues without exposing details to the caller.
    """

    def __init__(self, is_cn: bool = False) -> None:
        self._is_cn = is_cn

    def resolve(self, token: str) -> UserContext | None:
        try:
            client = _cached_garmin_client(token, self._is_cn)
        except Exception as exc:
            print(
                f"[garmin-mcp] Auth failed: {exc.__class__.__name__}: {str(exc)[:120]}",
                file=sys.stderr,
            )
            return None

        user_id = hashlib.sha256(token.encode()).hexdigest()[:16]
        return UserContext(user_id=user_id, garmin_client=client)
