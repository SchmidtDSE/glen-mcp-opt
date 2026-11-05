import json

import pytest
from starlette.testclient import TestClient

from glen_opt.server import create_app, mcp

pytestmark = pytest.mark.anyio


async def test_tools_listed():
    names = {t.name for t in await mcp.list_tools()}
    assert names == {"modeling_guide", "check_model", "solve_model", "list_solvers"}


async def test_check_model_accepts_object_and_string():
    spec = {"variables": {"x": {"ub": 3}}, "objective": {"sense": "max", "expr": "x"}}
    for arg in (spec, json.dumps(spec)):
        r = await mcp.call_tool("check_model", {"spec": arg})
        out = json.loads(r.content[0].text)
        assert out["ok"] and out["class"] == "LP" and out["solvable"]


async def test_solve_model_tool():
    spec = {"variables": {"x": {"ub": 3}}, "objective": {"sense": "max", "expr": "2*x"}}
    r = await mcp.call_tool("solve_model", {"spec": spec, "time_limit": 5})
    out = json.loads(r.content[0].text)
    assert out["status"] == "optimal" and out["objective"] == 6


async def test_modeling_guide():
    r = await mcp.call_tool("modeling_guide", {})
    assert "## Patterns" in r.content[0].text


def test_healthz():
    with TestClient(create_app()) as c:
        assert c.get("/healthz").text == "ok"
