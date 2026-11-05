"""HiGHS backend: LP, ILP, MILP."""

from __future__ import annotations

import math
import time

import highspy
import numpy as np

from .base import Solution, SolverBackend

_INF = highspy.kHighsInf


def _inf(v: float) -> float:
    return _INF if v == math.inf else (-_INF if v == -math.inf else v)


class HighsBackend(SolverBackend):
    name = "highs"
    classes = frozenset({"LP", "ILP", "MILP"})
    global_optimal = True

    def solve(self, model, time_limit: float, mip_gap: float | None) -> Solution:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(time_limit))
        if mip_gap is not None:
            h.setOptionValue("mip_rel_gap", float(mip_gap))

        n = len(model.col_names)
        h.addVars(n, np.array([_inf(v) for v in model.lb], dtype=float),
                  np.array([_inf(v) for v in model.ub], dtype=float))
        cols = np.arange(n, dtype=np.int32)
        cost = np.zeros(n)
        for col, c in model.objective.coefs.items():
            cost[col] = c
        h.changeColsCost(n, cols, cost)
        if model.sense == "max":
            h.changeObjectiveSense(highspy.ObjSense.kMaximize)
        if any(model.integer):
            idx = np.array([i for i, b in enumerate(model.integer) if b], dtype=np.int32)
            h.changeColsIntegrality(len(idx), idx, np.array([highspy.HighsVarType.kInteger] * len(idx)))

        rows = model.rows
        if rows:
            starts, index, value = [], [], []
            for r in rows:
                starts.append(len(index))
                index.extend(r.coefs.keys())
                value.extend(r.coefs.values())
            h.addRows(len(rows), np.array([_inf(r.lo) for r in rows], dtype=float),
                      np.array([_inf(r.hi) for r in rows], dtype=float), len(index),
                      np.array(starts, dtype=np.int32), np.array(index, dtype=np.int32),
                      np.array(value, dtype=float))

        t0 = time.monotonic()
        h.run()
        seconds = time.monotonic() - t0

        ms = h.getModelStatus()
        info = h.getInfo()
        S = highspy.HighsModelStatus
        status = {
            S.kOptimal: "optimal", S.kInfeasible: "infeasible", S.kUnbounded: "unbounded",
            S.kUnboundedOrInfeasible: "infeasible_or_unbounded", S.kTimeLimit: "time_limit",
            S.kModelEmpty: "optimal",
        }.get(ms, "error")
        has_solution = info.primal_solution_status == 2  # feasible
        sol = Solution(status, has_solution, solve_seconds=round(seconds, 3),
                       message="" if status != "error" else h.modelStatusToString(ms))
        offset = model.objective.const
        if has_solution:
            s = h.getSolution()
            sol.x = list(s.col_value)
            sol.row_activity = list(s.row_value)
            sol.objective = info.objective_function_value + offset
            if any(model.integer):
                sol.bound = info.mip_dual_bound + offset
                sol.gap = info.mip_gap
            elif s.dual_valid:
                sol.row_dual = list(s.row_dual)
        if status == "infeasible" and rows:
            try:
                h.setOptionValue("iis_strategy", 2)
                st, iis = h.getIis()
                if st == highspy.HighsStatus.kOk and iis.valid_:
                    sol.conflict_rows = list(iis.row_index_)
            except Exception:
                pass
        return sol
