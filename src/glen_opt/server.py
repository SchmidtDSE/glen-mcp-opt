"""MCP server for optimization in GLEN. Outline: tools are stubs."""

from __future__ import annotations

import json
import os

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.cors import CORSMiddleware

from . import solvers

mcp = MCPServer(name="glen-mcp-opt")


def _todo(tool: str) -> str:
    return json.dumps({"status": "not_implemented", "message": f"{tool} is not implemented yet"})


@mcp.tool(description="Choose parcels to maximize a benefit within a budget. TODO: define inputs.")
async def select_parcels() -> str:
    # TODO: inputs (candidates, benefit, cost, budget, connectivity?)
    # TODO: build the model and call a solver backend
    return _todo("select_parcels")


@mcp.tool(description="List available solvers.")
async def list_solvers() -> str:
    return json.dumps({"solvers": [b.name for b in solvers.available()]})


def create_app():
    app = mcp.streamable_http_app(
        stateless_http=True, json_response=True, host="0.0.0.0",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    # GLEN calls MCP servers from the browser, so CORS is needed.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["mcp-session-id"])
    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
