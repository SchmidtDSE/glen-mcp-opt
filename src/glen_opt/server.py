"""MCP server: turn optimization questions into models and solve them."""

from __future__ import annotations

import asyncio
import json
import os
from importlib.resources import files

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse

from . import solvers
from .model import Model, ModelErrors, build, eval_reports

MAX_TIME_LIMIT = float(os.environ.get("MAX_TIME_LIMIT", 300))
MAX_VALUES = 500
MAX_BINDING = 100

GUIDE = files("glen_opt").joinpath("guide.md").read_text()

mcp = MCPServer(name="glen-mcp-opt")


def _dump(d: dict) -> str:
    return json.dumps(d, default=str)


def _errors(e: ModelErrors) -> dict:
    return {"ok": False, "errors": [x.as_dict() for x in e.errors],
            "next": "Fix these errors in the spec and call check_model again."}


def _nonlinear(m: Model) -> list[dict]:
    return [e.as_dict() for e in m.nonlinear]


def check(spec) -> dict:
    try:
        m = build(spec)
    except ModelErrors as e:
        return _errors(e)
    backends = solvers.for_class(m.model_class)
    out = {"ok": True, "class": m.model_class, "size": m.size(),
           "solvers": [b.name for b in backends], "solvable": bool(backends)}
    if m.nonlinear:
        out["nonlinear_terms"] = _nonlinear(m)
    if not backends:
        out["next"] = (f"No available solver handles {m.model_class}. Linearize exactly if possible, otherwise "
                       "offer the user an approximation (see modeling_guide).")
    if m.warnings:
        out["warnings"] = m.warnings
    out["model"] = m.summary
    return out


def _value(v: float, integer: bool):
    if integer or abs(v - round(v)) < 1e-9:
        return int(round(v))
    return round(v, 6)


def _status_message(status: str, gap, has_solution: bool, sense) -> str:
    if status == "optimal":
        return "Optimal: no better solution exists." if sense else "Feasible solution found."
    if status == "time_limit":
        if not has_solution:
            return "Stopped at the time limit without finding any feasible solution."
        if gap is not None:
            return (f"Stopped at the time limit. The best solution found is within {gap:.2%} "
                    "of the best possible objective.")
        return "Stopped at the time limit; the solution may not be optimal."
    if status == "infeasible":
        return "No solution satisfies all constraints."
    if status == "unbounded":
        return "The objective can improve without limit; a constraint or bound is probably missing."
    if status == "infeasible_or_unbounded":
        return "The model is infeasible or unbounded."
    return "The solver failed."


def solve(spec, time_limit: float = 60, mip_gap: float | None = None, solver: str | None = None) -> dict:
    try:
        m = build(spec)
    except ModelErrors as e:
        return _errors(e)
    cls = m.model_class
    if solver:
        backend = solvers.get(solver)
        if backend is None:
            return {"ok": False, "errors": [{"where": "solver", "message": f"unknown solver '{solver}'"}]}
        if cls not in backend.classes:
            return {"ok": False, "errors": [{"where": "solver", "message": f"{solver} does not handle {cls}"}]}
    else:
        candidates = solvers.for_class(cls)
        if not candidates:
            return {"ok": True, "status": "unsupported", "class": cls, "nonlinear_terms": _nonlinear(m),
                    "message": f"No available solver handles {cls}.",
                    "next": "Linearize exactly if possible, otherwise offer the user an approximation (see modeling_guide)."}
        backend = candidates[0]

    time_limit = max(1.0, min(float(time_limit), MAX_TIME_LIMIT))
    s = backend.solve(m, time_limit, mip_gap)
    out = {"ok": True, "status": s.status, "message": _status_message(s.status, s.gap, s.has_solution, m.sense),
           "class": cls, "solver": backend.name, "solve_seconds": s.solve_seconds, "size": m.size()}
    if s.message:
        out["solver_message"] = s.message
    if s.has_solution:
        if m.sense:
            out["objective"] = _value(s.objective, False)
        if s.gap is not None and s.status != "optimal":
            out["bound"] = _value(s.bound, False)
            out["gap"] = round(s.gap, 6)
        if not backend.global_optimal:
            out["note"] = "Local solver: a better solution may exist."
        out.update(_variables(m, s.x))
        out.update(_binding(m, s.row_activity, s.row_dual))
        if m.reports:
            out["report"] = eval_reports(m, s.x)
    if s.conflict_rows:
        out["conflicting_constraints"] = [m.rows[i].name for i in s.conflict_rows]
    if m.warnings:
        out["warnings"] = m.warnings
    return out


def _variables(m: Model, x: list[float]) -> dict:
    values: dict = {}
    shown = nonzero = 0
    for col, v in enumerate(x):
        if abs(v) < 1e-9:
            continue
        nonzero += 1
        if shown >= MAX_VALUES:
            continue
        name, key = m.col_var[col]
        val = _value(v, m.integer[col])
        if key:
            values.setdefault(name, {})[", ".join(key)] = val
        else:
            values[name] = val
        shown += 1
    out = {"variables": values, "nonzero_count": nonzero}
    if nonzero > shown:
        out["variables_truncated"] = f"showing {shown} of {nonzero} nonzero values; add `report` entries for totals"
    return out


def _binding(m: Model, activity: list[float], dual: list[float] | None) -> dict:
    binding = []
    for i, (r, a) in enumerate(zip(m.rows, activity)):
        tol = 1e-6 * (1 + abs(a))
        if (r.hi != float("inf") and abs(a - r.hi) <= tol) or (r.lo != float("-inf") and abs(a - r.lo) <= tol):
            entry = {"name": r.name}
            if dual is not None:
                entry["dual"] = round(dual[i], 6)
            binding.append(entry)
    out = {"binding_constraints": binding[:MAX_BINDING]}
    if len(binding) > MAX_BINDING:
        out["binding_truncated"] = f"showing {MAX_BINDING} of {len(binding)}"
    if dual is not None and binding:
        out["dual_meaning"] = "dual = change in the objective per unit increase of the constraint's right-hand side"
    return out


# ---- tools ---------------------------------------------------------------

SPEC_HINT = ('Spec example: {"sets": {"A": ["a1", "a2"]}, "params": {"cost": {"index": "A", "values": {"a1": 3, "a2": 5}}, '
             '"budget": 6}, "variables": {"x": {"index": "A", "type": "binary"}}, "objective": {"sense": "max", '
             '"expr": "sum(x[a] for a in A)"}, "constraints": {"budget": {"expr": "sum(cost[a] * x[a] for a in A) <= budget"}}}')


@mcp.tool(description=(
    "Reference for writing optimization models: spec format, expression syntax, SQL data sources, "
    "modeling patterns (linearization, approximations, multiple objectives), and how to explain results. "
    "Read it before the first check_model call in a conversation."
))
async def modeling_guide() -> str:
    return GUIDE


@mcp.tool(description=(
    "Validate an optimization model without solving it. Returns errors to fix, or the model class "
    "(LP/ILP/MILP/NLP/...), size, whether an available solver handles it, and a plain-language summary. "
    "Ask the user for missing numbers (budget, targets) before formulating; never invent them. "
    "Call this before solve_model and fix every error. Show the user the summary for non-trivial models. "
    "spec: the model as a JSON object (see modeling_guide). " + SPEC_HINT
))
async def check_model(spec: dict | str) -> str:
    return _dump(await asyncio.to_thread(check, spec))


@mcp.tool(description=(
    "Solve an optimization model (same spec as check_model). Returns status, objective, nonzero variable values, "
    "binding constraints (with duals for LPs), `report` values, and conflicting constraints if infeasible. "
    f"time_limit in seconds (default 60, max {int(MAX_TIME_LIMIT)}); on timeout the best solution found is "
    "returned with its optimality gap. Explain results to the user as outcomes in their terms, not "
    "optimization jargon, and state solution quality (optimal, or within X% of optimal)."
))
async def solve_model(spec: dict | str, time_limit: float = 60, mip_gap: float | None = None,
                      solver: str | None = None) -> str:
    return _dump(await asyncio.to_thread(solve, spec, time_limit, mip_gap, solver))


@mcp.tool(description="List available solvers and the model classes each handles.")
async def list_solvers() -> str:
    return _dump({"solvers": [{"name": b.name, "classes": sorted(b.classes), "global_optimal": b.global_optimal}
                              for b in solvers.available()]})


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request):
    return PlainTextResponse("ok")


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
