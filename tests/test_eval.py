import importlib.util
import json
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.anyio

spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).parents[1] / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

FARM = {"variables": {"w": {}, "c": {}}, "objective": {"sense": "max", "expr": "40*w + 30*c"},
        "constraints": {"land": "w + c <= 100", "labor": "3*w + 2*c <= 240"}}


async def test_runner_with_mock_llm():
    replies = iter([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "solve_model", "arguments": json.dumps({"spec": FARM})}}]},
        {"role": "assistant", "content": "Maximum profit is $3,400."},
    ])

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": next(replies)}]})

    problem = json.loads((Path(__file__).parents[1] / "eval" / "problems.json").read_text())[0]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        r = await run_eval.run_one(client, "http://llm/chat/completions", "m", problem,
                                   await run_eval.tools_schema())
    assert r["pass"] and r["objective"] == 3400 and r["tool_calls"] == 1


def test_problems_have_expectations():
    problems = json.loads((Path(__file__).parents[1] / "eval" / "problems.json").read_text())
    assert len({p["id"] for p in problems}) == len(problems)
    assert all(("expected_objective" in p) != ("expected_status" in p) for p in problems)
