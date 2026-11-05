"""Solver backend interface.

A backend declares the model classes it solves and returns a `Solution`;
server.py turns that into the tool result, so every backend produces the same output.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Solution:
    status: str  # optimal | time_limit | infeasible | unbounded | infeasible_or_unbounded | error
    has_solution: bool
    objective: float | None = None
    bound: float | None = None
    gap: float | None = None
    x: list[float] = field(default_factory=list)
    row_activity: list[float] = field(default_factory=list)
    row_dual: list[float] | None = None
    conflict_rows: list[int] | None = None
    solve_seconds: float = 0.0
    message: str = ""


class SolverBackend:
    name: str = ""
    classes: frozenset[str] = frozenset()
    global_optimal: bool = True

    def available(self) -> bool:
        return True

    def solve(self, model, time_limit: float, mip_gap: float | None) -> Solution:
        raise NotImplementedError
