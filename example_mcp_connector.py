"""Minimal example: call garmin-mcp via the Anthropic MCP connector.

Prerequisites
-------------
1. garmin-mcp-auth  (run once locally to generate ~/.garminconnect_base64)
2. A running garmin-mcp server:
       GARMIN_MCP_TRANSPORT=streamable-http garmin-mcp
3. pip install anthropic

Usage
-----
   export ANTHROPIC_API_KEY=sk-ant-...
   export GARMIN_TOKEN=$(cat ~/.garminconnect_base64)
   export GARMIN_MCP_URL=http://localhost:8000/mcp   # or your deployed URL
   python example_mcp_connector.py
"""

import os
import anthropic

GARMIN_TOKEN = os.environ["GARMIN_TOKEN"]       # base64 Garmin OAuth token
GARMIN_MCP_URL = os.environ.get("GARMIN_MCP_URL", "http://localhost:8000/mcp")

client = anthropic.Anthropic()

response = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "What were my last 3 activities? Show sport type, date, and duration.",
        }
    ],
    mcp_servers=[
        {
            "type": "url",
            "url": GARMIN_MCP_URL,
            "name": "garmin",
            "authorization_token": GARMIN_TOKEN,
        }
    ],
    tools=[
        {
            "type": "mcp_toolset",
            "mcp_server_name": "garmin",
        }
    ],
    betas=["mcp-client-2025-11-20"],
)

for block in response.content:
    if hasattr(block, "text"):
        print(block.text)
