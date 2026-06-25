"""
Modular MCP Server for Garmin Connect Data
"""

import os
import sys
import base64

import requests
from mcp.server.fastmcp import FastMCP

from garminconnect import Garmin, GarminConnectAuthenticationError, GarminConnectConnectionError, GarminConnectTooManyRequestsError

# Import all modules
from garmin_mcp import activity_management
from garmin_mcp import health_wellness
from garmin_mcp import user_profile
from garmin_mcp import devices
from garmin_mcp import gear_management
from garmin_mcp import weight_management
from garmin_mcp import challenges
from garmin_mcp import training
from garmin_mcp import workouts
from garmin_mcp import workout_templates
from garmin_mcp import data_management
from garmin_mcp import womens_health
from garmin_mcp import nutrition
from garmin_mcp import workout_builders
from garmin_mcp import courses
from garmin_mcp import activity_analysis
from garmin_mcp.auth import GarminTokenAuthProvider
from garmin_mcp.middleware import GarminAuthMiddleware


def is_interactive_terminal() -> bool:
    """Detect if running in interactive terminal vs MCP subprocess.

    Returns:
        bool: True if running in an interactive terminal, False otherwise
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def get_mfa() -> str:
    """Get MFA code from user input.

    Raises:
        RuntimeError: If running in non-interactive environment
    """
    if not is_interactive_terminal():
        print(
            "\nERROR: MFA code required but no interactive terminal available.\n"
            "Please run 'garmin-mcp-auth' in your terminal first.\n"
            "See: https://github.com/Taxuspt/garmin_mcp#mfa-setup\n",
            file=sys.stderr,
        )
        raise RuntimeError("MFA required but non-interactive environment")

    print(
        "\nGarmin Connect MFA required. Please check your email/phone for the code.",
        file=sys.stderr,
    )
    return input("Enter MFA code: ")


# Get credentials from environment
email = os.environ.get("GARMIN_EMAIL")
email_file = os.environ.get("GARMIN_EMAIL_FILE")
if email and email_file:
    raise ValueError(
        "Must only provide one of GARMIN_EMAIL and GARMIN_EMAIL_FILE, got both"
    )
elif email_file:
    with open(email_file, "r") as email_file:
        email = email_file.read().rstrip()

password = os.environ.get("GARMIN_PASSWORD")
password_file = os.environ.get("GARMIN_PASSWORD_FILE")
if password and password_file:
    raise ValueError(
        "Must only provide one of GARMIN_PASSWORD and GARMIN_PASSWORD_FILE, got both"
    )
elif password_file:
    with open(password_file, "r") as password_file:
        password = password_file.read().rstrip()

tokenstore = os.getenv("GARMINTOKENS") or "~/.garminconnect"
tokenstore_base64 = os.getenv("GARMINTOKENS_BASE64") or "~/.garminconnect_base64"
is_cn = os.getenv("GARMIN_IS_CN", "false").lower() in ("true", "1", "yes")


# --- Tool filtering ---------------------------------------------------------
# Optionally expose only a subset of tools, to reduce the context an LLM must
# carry. No modules are removed; tools are simply not registered when filtered.
#   GARMIN_ENABLED_TOOLS  - comma-separated allowlist; if set, ONLY these register
#   GARMIN_DISABLED_TOOLS - comma-separated denylist; ignored if an allowlist is set
# Tool names are case-insensitive. Unset = all tools register (default behaviour).
def _parse_tool_set(value):
    if not value:
        return set()
    return {name.strip().lower() for name in value.split(",") if name.strip()}


enabled_tools = _parse_tool_set(os.getenv("GARMIN_ENABLED_TOOLS"))
disabled_tools = _parse_tool_set(os.getenv("GARMIN_DISABLED_TOOLS"))


_VALID_TRANSPORTS = ("stdio", "streamable-http", "sse")


class _GarminProxy:
    """Wraps the Garmin client to translate known runtime exceptions into clear messages.

    Without this, token expiry or rate-limiting during a tool call surfaces raw
    library tracebacks to the MCP client. The proxy intercepts each attribute
    access and, if the result is callable, wraps the call so that known Garmin
    exceptions become user-friendly strings rather than server errors.
    """

    _MESSAGES = {
        GarminConnectAuthenticationError: (
            "Garmin authentication expired. "
            "Re-run 'garmin-mcp-auth' to refresh your tokens and restart the server."
        ),
        GarminConnectTooManyRequestsError: (
            "Garmin rate limit hit. Wait a few minutes before retrying."
        ),
        GarminConnectConnectionError: (
            "Garmin Connect is unreachable. Check your network connection or try again later."
        ),
    }

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def _call(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except tuple(self._MESSAGES) as exc:
                for exc_type, msg in self._MESSAGES.items():
                    if isinstance(exc, exc_type):
                        raise type(exc)(msg) from None
                raise

        return _call


def _parse_transport_config() -> tuple[str, str, int]:
    """Read and validate HTTP transport env vars. Raises ValueError on bad input."""
    transport = os.getenv("GARMIN_MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in _VALID_TRANSPORTS:
        raise ValueError(
            f"Invalid GARMIN_MCP_TRANSPORT {transport!r}; "
            f"expected one of {', '.join(_VALID_TRANSPORTS)}"
        )
    http_host = os.getenv("GARMIN_MCP_HOST", "0.0.0.0")
    http_port = int(os.getenv("GARMIN_MCP_PORT", "8000"))
    return transport, http_host, http_port


class _ToolFilter:
    """Wraps a FastMCP app to conditionally register tools by function name.

    Modules register via ``@app.tool()``; we intercept that decorator and skip
    registration for any tool not permitted by the env-var filter. All other
    attribute access (``run``, ``resource``, ...) passes through to the app.
    """

    def __init__(self, app, enabled, disabled):
        self._app = app
        self._enabled = enabled
        self._disabled = disabled
        self._seen = set()  # tool names encountered, for typo detection

    def _allowed(self, name):
        name = name.lower()
        if self._enabled:
            return name in self._enabled
        return name not in self._disabled

    def tool(self, *args, **kwargs):
        decorator = self._app.tool(*args, **kwargs)
        # Prefer the explicit registered name if given (@app.tool(name="x")),
        # so the env-var filter matches what the user actually configures.
        explicit = kwargs.get("name") or (
            args[0] if args and isinstance(args[0], str) else None
        )

        def wrapper(fn):
            name = explicit or getattr(fn, "__name__", "")
            self._seen.add(name.lower())
            if self._allowed(name):
                return decorator(fn)
            return fn  # skip registration; tool never reaches the LLM

        return wrapper

    def unknown_filter_names(self):
        """Configured names that never matched a real tool (likely typos)."""
        configured = self._enabled or self._disabled
        return sorted(configured - self._seen)

    def __getattr__(self, item):
        return getattr(self._app, item)
# ---------------------------------------------------------------------------


def _configure_all_modules(client) -> None:
    """Push a resolved Garmin client into every module's global.

    Called once per unique user on first request (Phase 1: single user, fires
    exactly once).  Phase 2 upgrade: replace module globals with a ContextVar
    and update this function (or remove it) — the AuthProvider and middleware
    interfaces stay unchanged.
    """
    activity_management.configure(client)
    health_wellness.configure(client)
    user_profile.configure(client)
    devices.configure(client)
    gear_management.configure(client)
    weight_management.configure(client)
    challenges.configure(client)
    training.configure(client)
    workouts.configure(client)
    data_management.configure(client)
    womens_health.configure(client)
    nutrition.configure(client)
    workout_builders.configure(client)
    courses.configure(client)
    activity_analysis.configure(client)


def init_api(email, password):
    """Initialize Garmin API with your credentials."""
    import io

    try:
        # Using Oauth1 and OAuth2 token files from directory
        print(
            f"Trying to login to Garmin Connect using token data from directory '{tokenstore}'...\n",
            file=sys.stderr,
        )

        # Using Oauth1 and Oauth2 tokens from base64 encoded string
        # print(
        #     f"Trying to login to Garmin Connect using token data from file '{tokenstore_base64}'...\n"
        # )
        # dir_path = os.path.expanduser(tokenstore_base64)
        # with open(dir_path, "r") as token_file:
        #     tokenstore = token_file.read()

        # Suppress stderr for token validation to avoid confusing library errors
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            garmin = Garmin(is_cn=is_cn)
            garmin.login(tokenstore)
        finally:
            sys.stderr = old_stderr

    except (FileNotFoundError, GarminConnectConnectionError, GarminConnectTooManyRequestsError, GarminConnectAuthenticationError):
        # Session is expired. You'll need to log in again

        # Check if we're in a non-interactive environment without credentials
        if not is_interactive_terminal() and (not email or not password):
            print(
                "ERROR: OAuth tokens not found and no interactive terminal available.\n"
                "Please authenticate first:\n"
                "  1. Run: garmin-mcp-auth\n"
                "  2. Enter your credentials and MFA code\n"
                "  3. Restart your MCP client\n"
                f"Tokens will be saved to: {tokenstore}\n",
                file=sys.stderr,
            )
            return None

        print(
            "Login tokens not present, login with your Garmin Connect credentials to generate them.\n"
            f"They will be stored in '{tokenstore}' for future use.\n",
            file=sys.stderr,
        )
        try:
            garmin = Garmin(
                email=email, password=password, is_cn=is_cn, prompt_mfa=get_mfa, return_on_mfa=True
            )
            result1, result2 = garmin.login()
            if result1 == "needs_mfa":
                mfa_code = get_mfa()
                garmin.resume_login(result2, mfa_code)
            # Save Oauth1 and Oauth2 token files to directory for next login
            garmin.client.dump(tokenstore)
            print(
                f"Oauth tokens stored in '{tokenstore}' directory for future use. (first method)\n",
                file=sys.stderr,
            )
            # Encode Oauth1 and Oauth2 tokens to base64 string and save to file for next login (alternative way)
            expanded_tokenstore = os.path.expanduser(tokenstore)
            token_json_path = os.path.join(expanded_tokenstore, "garmin_tokens.json")
            with open(token_json_path, "r") as f:
                token_data = f.read()
            token_base64 = base64.b64encode(token_data.encode()).decode()
            dir_path = os.path.expanduser(tokenstore_base64)
            with open(dir_path, "w") as token_file:
                token_file.write(token_base64)
            print(
                f"Oauth tokens encoded as base64 string and saved to '{dir_path}' file for future use. (second method)\n",
                file=sys.stderr,
            )
        except (
            FileNotFoundError,
            GarminConnectConnectionError,
            GarminConnectTooManyRequestsError,
            GarminConnectAuthenticationError,
            requests.exceptions.HTTPError,
        ) as err:
            error_msg = str(err)

            # Provide clean, actionable error messages
            print("\nAuthentication failed.", file=sys.stderr)

            if isinstance(err, GarminConnectAuthenticationError):
                if "MFA" in error_msg or "code" in error_msg.lower():
                    print("MFA code may be incorrect or expired.", file=sys.stderr)
                else:
                    print("Invalid email or password.", file=sys.stderr)
            elif isinstance(err, GarminConnectTooManyRequestsError):
                print(
                    "Too many requests. Please wait and try again.", file=sys.stderr
                )
            elif isinstance(err, GarminConnectConnectionError):
                if "401" in error_msg or "Unauthorized" in error_msg:
                    print(
                        "Invalid credentials. Please check your email and password.",
                        file=sys.stderr,
                    )
                elif "500" in error_msg or "503" in error_msg:
                    print(
                        "Garmin Connect service issue. Please try again later.",
                        file=sys.stderr,
                    )
                else:
                    print(f"Error: {error_msg.split(':')[0]}", file=sys.stderr)
            elif isinstance(err, requests.exceptions.HTTPError):
                print("Network error. Please check your connection.", file=sys.stderr)
            else:
                print(f"Error: {error_msg.split(':')[0]}", file=sys.stderr)

            print(
                f"\nTip: Run 'garmin-mcp-auth' to authenticate interactively.",
                file=sys.stderr,
            )
            return None

    return garmin


def main():
    """Initialize the MCP server and register all tools"""

    # On Windows, stdout runs in text mode and translates \n to \r\n, which
    # breaks the MCP stdio framing that Claude Desktop and other clients expect.
    # Force binary-transparent newlines so JSON messages arrive intact.
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, newline="\n")

    # --- Transport configuration --------------------------------------------
    # By default the server speaks stdio (Claude Desktop, MCP Inspector, etc.).
    # Set GARMIN_MCP_TRANSPORT=streamable-http (or sse) to serve over HTTP.
    #   GARMIN_MCP_TRANSPORT - stdio (default) | streamable-http | sse
    #   GARMIN_MCP_HOST      - bind address for HTTP transports (default 0.0.0.0)
    #   GARMIN_MCP_PORT      - bind port for HTTP transports (default 8000)
    try:
        transport, http_host, http_port = _parse_transport_config()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    # --- stdio path (unchanged) ------------------------------------------------
    # stdio is used by Claude Desktop and MCP Inspector.  Garmin client is
    # initialised once at startup from local token files or credentials.
    if transport == "stdio":
        garmin_client = init_api(email, password)
        if not garmin_client:
            print("Failed to initialize Garmin Connect client. Exiting.", file=sys.stderr)
            return
        print("Garmin Connect client initialized successfully.", file=sys.stderr)
        garmin_client = _GarminProxy(garmin_client)
        _configure_all_modules(garmin_client)

    # --- HTTP path (MCP connector) --------------------------------------------
    # For streamable-http and sse the bearer token sent by the Claude API IS
    # the user's Garmin OAuth token (base64).  No local token storage needed;
    # the server is stateless.  Garmin client is resolved per-user on the first
    # request and cached (LRU) for the lifetime of the process.
    # Credentials / token-file env vars are intentionally ignored here.

    # Create the MCP app, wrapped so the env-var filter can drop tools.
    # host/port only matter for the HTTP transports; stdio ignores them.
    fastmcp = FastMCP("Garmin Connect v1.0", host=http_host, port=http_port)
    app = _ToolFilter(fastmcp, enabled_tools, disabled_tools)
    if enabled_tools:
        print(f"Tool filter: allowlist of {len(enabled_tools)} tool(s).", file=sys.stderr)
    elif disabled_tools:
        print(f"Tool filter: denylist of {len(disabled_tools)} tool(s).", file=sys.stderr)

    # Register tools from all modules
    app = activity_management.register_tools(app)
    app = health_wellness.register_tools(app)
    app = user_profile.register_tools(app)
    app = devices.register_tools(app)
    app = gear_management.register_tools(app)
    app = weight_management.register_tools(app)
    app = challenges.register_tools(app)
    app = training.register_tools(app)
    app = workouts.register_tools(app)
    app = data_management.register_tools(app)
    app = womens_health.register_tools(app)
    app = nutrition.register_tools(app)
    app = workout_builders.register_tools(app)
    app = courses.register_tools(app)
    app = activity_analysis.register_tools(app)

    # Register resources (workout templates)
    app = workout_templates.register_resources(app)

    # Warn about filter entries that matched no tool (most likely typos)
    unknown = app.unknown_filter_names()
    if unknown:
        print(
            f"Tool filter: warning — name(s) not found and ignored: {', '.join(unknown)}",
            file=sys.stderr,
        )

    if transport == "stdio":
        app.run(transport="stdio")
        return

    # HTTP transports — OAuth 2.0 Authorization Code + PKCE, then MCP with bearer auth.
    import base64 as _b64
    import hashlib as _hashlib
    import secrets as _secrets
    import time as _time

    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

    # In-memory auth-code store (single process, 5-minute TTL).
    # Each entry: {token, challenge, expires}
    _code_store: dict = {}

    def _issue_code(garmin_token: str, code_challenge: str) -> str:
        code = _secrets.token_urlsafe(32)
        _code_store[code] = {
            "token": garmin_token,
            "challenge": code_challenge,
            "expires": _time.time() + 300,
        }
        return code

    def _redeem_code(code: str, code_verifier: str) -> str | None:
        entry = _code_store.pop(code, None)
        if not entry or _time.time() > entry["expires"]:
            return None
        digest = _hashlib.sha256(code_verifier.encode()).digest()
        expected = _b64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if expected != entry["challenge"]:
            return None
        return entry["token"]

    # Create auth provider before the route closures that reference it.
    auth_provider = GarminTokenAuthProvider(is_cn=is_cn)

    _AUTHORIZE_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Connect Garmin to Claude</title>
  <style>
    *{{box-sizing:border-box}}
    body{{font-family:system-ui,sans-serif;max-width:460px;margin:80px auto;padding:0 24px;color:#111}}
    h1{{font-size:1.25rem;font-weight:700;margin-bottom:6px}}
    p{{color:#555;font-size:.9rem;line-height:1.5;margin:0 0 12px}}
    code{{background:#f3f4f6;padding:2px 6px;border-radius:3px;font-size:.78rem;word-break:break-all}}
    label{{display:block;font-size:.85rem;font-weight:600;margin:16px 0 6px}}
    textarea{{width:100%;height:96px;font-family:monospace;font-size:.72rem;border:1px solid #d1d5db;border-radius:4px;padding:8px;resize:vertical}}
    button{{margin-top:14px;background:#000;color:#fff;border:none;padding:12px 0;border-radius:4px;font-size:.95rem;cursor:pointer;width:100%}}
    .err{{color:#dc2626;font-size:.85rem;margin-top:8px}}
  </style>
</head>
<body>
  <h1>Connect Garmin to Claude</h1>
  <p>Paste your Garmin base64 token to let Claude read your training data.</p>
  <p>Generate it on your Mac:<br>
  <code>python3 -c "print(open('$HOME/.garminconnect_base64').read().strip())" | pbcopy</code></p>
  <form method="POST">
    <input type="hidden" name="state" value="{state}">
    <input type="hidden" name="redirect_uri" value="{redirect_uri}">
    <input type="hidden" name="code_challenge" value="{code_challenge}">
    <input type="hidden" name="code_challenge_method" value="{code_challenge_method}">
    <input type="hidden" name="client_id" value="{client_id}">
    <label for="t">Garmin token</label>
    <textarea id="t" name="garmin_token" placeholder="eyJ..." required></textarea>
    {error}
    <button type="submit">Authorize</button>
  </form>
</body>
</html>"""

    @fastmcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: "Request") -> "PlainTextResponse":
        return PlainTextResponse("ok")

    def _public_base(request: "Request") -> str:
        """Reconstruct the public-facing base URL.

        Fly.io (and most reverse proxies) terminate TLS at the edge and forward
        plain HTTP to the container, so request.base_url has scheme=http even
        though the public URL is https.  X-Forwarded-Proto carries the real scheme.
        """
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("x-forwarded-host", request.url.netloc)
        return f"{scheme}://{host}"

    @fastmcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
    async def oauth_metadata(request: "Request") -> "JSONResponse":
        base = _public_base(request)
        return JSONResponse({
            "issuer": base,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/oauth/token",
            "registration_endpoint": f"{base}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "code_challenge_methods_supported": ["S256"],
        })

    @fastmcp.custom_route("/register", methods=["POST"])
    async def register(request: "Request") -> "JSONResponse":
        """RFC 7591 Dynamic Client Registration — accept any client, no stored state.

        We don't validate or store client metadata; the Garmin token entered at
        /authorize is the real credential.  We just hand back a client_id so
        claude.ai can proceed with the Authorization Code flow.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}
        client_id = body.get("client_id") or _secrets.token_urlsafe(16)
        return JSONResponse(
            {
                "client_id": client_id,
                "client_id_issued_at": int(_time.time()),
                "redirect_uris": body.get("redirect_uris", []),
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            status_code=201,
        )

    @fastmcp.custom_route("/authorize", methods=["GET", "POST"])
    async def authorize(request: "Request") -> "HTMLResponse | RedirectResponse":  # noqa: F811
        if request.method == "GET":
            p = request.query_params
            html = _AUTHORIZE_HTML.format(
                state=p.get("state", ""),
                redirect_uri=p.get("redirect_uri", ""),
                code_challenge=p.get("code_challenge", ""),
                code_challenge_method=p.get("code_challenge_method", "S256"),
                client_id=p.get("client_id", ""),
                error="",
            )
            return HTMLResponse(html)

        # POST — validate token, issue code, redirect back to claude.ai
        form = await request.form()
        state = form.get("state", "")
        redirect_uri = form.get("redirect_uri", "")
        code_challenge = form.get("code_challenge", "")
        code_challenge_method = form.get("code_challenge_method", "S256")
        client_id = form.get("client_id", "")
        garmin_token = (form.get("garmin_token") or "").strip()

        ctx = auth_provider.resolve(garmin_token)
        if ctx is None:
            html = _AUTHORIZE_HTML.format(
                state=state, redirect_uri=redirect_uri,
                code_challenge=code_challenge, code_challenge_method=code_challenge_method,
                client_id=client_id,
                error='<p class="err">Invalid token — check your Garmin base64 token and try again.</p>',
            )
            return HTMLResponse(html, status_code=400)

        code = _issue_code(garmin_token, code_challenge)
        sep = "&" if "?" in redirect_uri else "?"
        return RedirectResponse(f"{redirect_uri}{sep}code={code}&state={state}", status_code=302)

    @fastmcp.custom_route("/oauth/token", methods=["POST"])
    async def oauth_token(request: "Request") -> "JSONResponse":
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            grant_type = body.get("grant_type", "")
            code = body.get("code", "")
            code_verifier = body.get("code_verifier", "")
        else:
            form = await request.form()
            grant_type = form.get("grant_type", "")
            code = form.get("code", "")
            code_verifier = form.get("code_verifier", "")

        if grant_type != "authorization_code":
            return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

        garmin_token = _redeem_code(code, code_verifier)
        if garmin_token is None:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)

        return JSONResponse({
            "access_token": garmin_token,
            "token_type": "bearer",
            "expires_in": 3600,
        })

    def _on_new_client(raw_client) -> None:
        """Wrap and push a freshly resolved Garmin client into all modules."""
        _configure_all_modules(_GarminProxy(raw_client))

    if transport == "streamable-http":
        starlette_app = fastmcp.streamable_http_app()
    else:  # sse
        starlette_app = fastmcp.sse_app()

    wrapped_app = GarminAuthMiddleware(starlette_app, auth_provider, on_new_client=_on_new_client)

    print(
        f"Serving MCP over {transport} on {http_host}:{http_port} "
        "(bearer token = base64 Garmin OAuth token)",
        file=sys.stderr,
    )

    import anyio
    import uvicorn

    async def _serve() -> None:
        config = uvicorn.Config(
            wrapped_app,
            host=http_host,
            port=http_port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()

    anyio.run(_serve)


if __name__ == "__main__":
    main()
