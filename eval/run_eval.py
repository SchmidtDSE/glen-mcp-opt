"""Run eval problems through LLMs that use this server's tools.

Calls an OpenAI-compatible chat endpoint (open-llm-proxy) and executes tool calls in-process.

    PROXY_KEY=... python eval/run_eval.py                      # all models
    PROXY_KEY=... python eval/run_eval.py --models kimi qwen3  # selected models
    PROXY_KEY=... python eval/run_eval.py --problems farm_lp   # selected problems

Pass: the last successful solve_model result matches `expected_objective` (relative tolerance 1e-4)
or `expected_status`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx

from glen_opt.server import mcp

logging.getLogger("httpx").setLevel(logging.WARNING)

# The models tpl-ca offers (tpl-ca/k8s/configmap.yaml).
MODELS = [
    "xiaomi/mimo-v2.6-pro", "xiaomi/mimo-v2.6-flash", "z-ai/glm-5.2", "deepseek/deepseek-v4-flash",
    "qwen", "qwen3", "qwen3-small", "gpt-oss", "kimi", "minimax-m2", "anthropic/claude-sonnet-5",
]
SYSTEM = ("You answer optimization questions using the available tools. All data needed is in the question; "
          "do not ask the user for more information. Finish with a short answer that states the optimal value.")
MAX_TURNS = 15


async def tools_schema() -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.input_schema}}
            for t in await mcp.list_tools()]


async def run_one(client: httpx.AsyncClient, url: str, model: str, problem: dict, tools: list[dict]) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": problem["question"]}]
    calls: list[str] = []
    check_errors = 0
    last_solve = None
    t0 = time.monotonic()
    for _ in range(MAX_TURNS):
        resp = await client.post(url, json={"model": model, "messages": messages, "tools": tools})
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        if not msg.get("tool_calls"):
            break
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            calls.append(name)
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
                result = (await mcp.call_tool(name, args)).content[0].text
            except Exception as e:  # bad arguments, unknown tool
                result = json.dumps({"ok": False, "errors": [{"where": "call", "message": str(e)}]})
            out = json.loads(result) if result.startswith("{") else {}
            if name == "check_model" and not out.get("ok", True):
                check_errors += 1
            if name == "solve_model" and out.get("ok") and "status" in out:
                last_solve = out
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    return {"model": model, "problem": problem["id"], "pass": score(problem, last_solve),
            "objective": (last_solve or {}).get("objective"), "status": (last_solve or {}).get("status"),
            "tool_calls": len(calls), "check_errors": check_errors, "seconds": round(time.monotonic() - t0, 1),
            "final": (messages[-1].get("content") or "")[:300]}


def score(problem: dict, solve: dict | None) -> bool:
    if solve is None:
        return False
    if "expected_status" in problem:
        return solve.get("status") == problem["expected_status"]
    obj = solve.get("objective")
    exp = problem["expected_objective"]
    return obj is not None and solve.get("status") == "optimal" and abs(obj - exp) <= 1e-4 * max(1, abs(exp))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--problems", nargs="*")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://open-llm-proxy.nrp-nautilus.io/v1"))
    ap.add_argument("--out", default="eval/results.json")
    a = ap.parse_args()

    problems = json.loads((Path(__file__).parent / "problems.json").read_text())
    if a.problems:
        problems = [p for p in problems if p["id"] in a.problems]
    tools = await tools_schema()
    headers = {"Authorization": f"Bearer {os.environ['PROXY_KEY']}"}
    url = a.base_url.rstrip("/") + "/chat/completions"
    results = []
    async with httpx.AsyncClient(headers=headers, timeout=600) as client:
        for model in a.models:
            for p in problems:
                try:
                    r = await run_one(client, url, model, p, tools)
                except Exception as e:
                    r = {"model": model, "problem": p["id"], "pass": False, "error": str(e)[:300]}
                results.append(r)
                Path(a.out).write_text(json.dumps(results, indent=2))
                print(f"{model:32} {p['id']:20} {'PASS' if r['pass'] else 'FAIL'}"
                      f"  {r.get('error') or ''}", flush=True)

    print("\n| model | passed | avg tool calls | check errors |\n|---|---|---|---|")
    for model in a.models:
        rs = [r for r in results if r["model"] == model]
        n = sum(r["pass"] for r in rs)
        calls = sum(r.get("tool_calls", 0) for r in rs) / max(1, len(rs))
        errs = sum(r.get("check_errors", 0) for r in rs)
        print(f"| {model} | {n}/{len(rs)} | {calls:.1f} | {errs} |")


if __name__ == "__main__":
    asyncio.run(main())
