import itertools

from glen_opt.server import check, solve


def test_lp_with_duals():
    # max 40w + 30c, w + c <= 100 (land), 3w + 2c <= 240 (labor) -> w=40, c=60, profit 3400
    spec = {
        "variables": {"w": {}, "c": {}},
        "objective": {"sense": "max", "expr": "40*w + 30*c"},
        "constraints": {"land": "w + c <= 100", "labor": "3*w + 2*c <= 240"},
    }
    r = solve(spec)
    assert r["status"] == "optimal"
    assert r["objective"] == 3400
    assert r["variables"] == {"w": 40, "c": 60}
    duals = {b["name"]: b["dual"] for b in r["binding_constraints"]}
    assert duals == {"land": 10, "labor": 10}


def test_dual_is_per_unit_increase_of_rhs():
    # Land written as -w - c >= -100: raising the RHS to -99 removes an acre, so the dual is -10.
    spec = {
        "variables": {"w": {}, "c": {}},
        "objective": {"sense": "max", "expr": "40*w + 30*c"},
        "constraints": {"land": "-w - c >= -100", "labor": "3*w + 2*c <= 240"},
    }
    r = solve(spec)
    duals = {b["name"]: b["dual"] for b in r["binding_constraints"]}
    assert duals == {"land": -10, "labor": 10}


def _knapsack(cost, benefit, budget, ub):
    best = 0
    for x in itertools.product(*(range(u + 1) for u in ub)):
        if sum(c * v for c, v in zip(cost, x)) <= budget:
            best = max(best, sum(b * v for b, v in zip(benefit, x)))
    return best


def test_binary_knapsack_matches_brute_force():
    cost, benefit, budget = [3, 5, 2, 4, 6], [4, 7, 2, 5, 8], 10
    spec = {
        "sets": {"A": {"range": 5}},
        "params": {"cost": {"index": "A", "values": cost}, "benefit": {"index": "A", "values": benefit},
                   "budget": budget},
        "variables": {"x": {"index": "A", "type": "binary"}},
        "objective": {"sense": "max", "expr": "sum(benefit[a] * x[a] for a in A)"},
        "constraints": {"budget": "sum(cost[a] * x[a] for a in A) <= budget"},
        "report": {"spent": "sum(cost[a] * x[a] for a in A)"},
    }
    r = solve(spec)
    assert r["status"] == "optimal"
    assert r["objective"] == _knapsack(cost, benefit, budget, [1] * 5)
    assert r["report"]["spent"] <= budget


def test_general_integer_matches_brute_force():
    cost, benefit, ub, budget = [3, 5, 2], [4, 7, 2], [4, 1, 10], 20
    spec = {
        "sets": {"A": ["a1", "a2", "a3"]},
        "params": {"cost": {"index": "A", "values": cost}, "benefit": {"index": "A", "values": benefit},
                   "max_units": {"index": "A", "values": ub}, "budget": budget},
        "variables": {"x": {"index": "A", "type": "integer", "ub": "max_units"}},
        "objective": {"sense": "max", "expr": "sum(benefit[a] * x[a] for a in A)"},
        "constraints": {"budget": "sum(cost[a] * x[a] for a in A) <= budget"},
    }
    r = solve(spec)
    assert r["objective"] == _knapsack(cost, benefit, budget, ub)


def test_fixed_charge_milp():
    # Facilities A (fixed 100, cap 60, unit 2), B (fixed 80, cap 50, unit 3); demand 70.
    # Both must open: 180 + 60*2 + 10*3 = 330.
    spec = {
        "sets": {"F": ["A", "B"]},
        "params": {"fixed": {"index": "F", "values": [100, 80]}, "cap": {"index": "F", "values": [60, 50]},
                   "unit": {"index": "F", "values": [2, 3]}, "demand": 70},
        "variables": {"open": {"index": "F", "type": "binary"}, "y": {"index": "F"}},
        "objective": {"sense": "min", "expr": "sum(fixed[f] * open[f] + unit[f] * y[f] for f in F)"},
        "constraints": {"cap": {"for": "f in F", "expr": "y[f] <= cap[f] * open[f]"},
                        "demand": "sum(y[f] for f in F) >= demand"},
    }
    r = solve(spec)
    assert r["class"] == "MILP"
    assert r["objective"] == 330


def test_assignment_matches_brute_force():
    cost = {"r1": {"s1": 9, "s2": 2, "s3": 7}, "r2": {"s1": 6, "s2": 4, "s3": 3}, "r3": {"s1": 5, "s2": 8, "s3": 1}}
    R, S = ["r1", "r2", "r3"], ["s1", "s2", "s3"]
    best = min(sum(cost[r][s] for r, s in zip(R, p)) for p in itertools.permutations(S))
    spec = {
        "sets": {"R": R, "S": S},
        "params": {"cost": {"index": ["R", "S"], "values": cost}},
        "variables": {"x": {"index": ["R", "S"], "type": "binary"}},
        "objective": {"sense": "min", "expr": "sum(cost[r, s] * x[r, s] for r in R for s in S)"},
        "constraints": {"one_site": {"for": "r in R", "expr": "sum(x[r, s] for s in S) == 1"},
                        "one_ranger": {"for": "s in S", "expr": "sum(x[r, s] for r in R) == 1"}},
    }
    r = solve(spec)
    assert r["objective"] == best


def test_infeasible_reports_conflict():
    spec = {
        "variables": {"x": {}, "y": {}},
        "objective": {"sense": "min", "expr": "x + y"},
        "constraints": {"cap": "x + y <= 5", "need": "x + y >= 10", "other": "x - y <= 100"},
    }
    r = solve(spec)
    assert r["status"] == "infeasible"
    assert sorted(r["conflicting_constraints"]) == ["cap", "need"]


def test_unbounded():
    r = solve({"variables": {"x": {}}, "objective": {"sense": "max", "expr": "x"}})
    assert r["status"] in ("unbounded", "infeasible_or_unbounded")


def test_feasibility_only():
    r = solve({"variables": {"x": {"type": "integer"}}, "constraints": {"c": "2 <= x <= 3"}})
    assert r["status"] == "optimal"
    assert "objective" not in r
    assert r["variables"]["x"] in (2, 3)


def test_nonlinear_is_unsupported():
    spec = {"variables": {"x": {}, "y": {}}, "objective": {"sense": "min", "expr": "x * y"}}
    c = check(spec)
    assert c["ok"] and not c["solvable"] and c["class"] == "NLP"
    r = solve(spec)
    assert r["status"] == "unsupported"
    assert "product of variables" in r["nonlinear_terms"][0]["message"]


def test_binary_product_linearized_matches_brute_force():
    # Choose nodes; benefit per node plus a bonus per chosen adjacent pair; budget on count.
    N, E = ["a", "b", "c", "d"], [["a", "b"], ["b", "c"], ["c", "d"]]
    w, bonus, k = {"a": 3, "b": 1, "c": 1, "d": 3}, 4, 2
    best = 0
    for combo in itertools.combinations(N, k):
        s = set(combo)
        best = max(best, sum(w[n] for n in s) + bonus * sum(1 for i, j in E if i in s and j in s))
    spec = {
        "sets": {"N": N, "E": E},
        "params": {"w": {"index": "N", "values": w}, "bonus": bonus, "k": k},
        "variables": {"x": {"index": "N", "type": "binary"}, "z": {"index": "E", "type": "binary"}},
        "objective": {"sense": "max", "expr": "sum(w[n] * x[n] for n in N) + bonus * sum(z[i, j] for i, j in E)"},
        "constraints": {"k": "sum(x[n] for n in N) <= k",
                        "z1": {"for": "i, j in E", "expr": "z[i, j] <= x[i]"},
                        "z2": {"for": "i, j in E", "expr": "z[i, j] <= x[j]"}},
    }
    r = solve(spec)
    assert r["objective"] == best


def test_time_limit_clamped_and_errors_returned():
    r = solve({"variables": {"x": {}}, "objective": {"sense": "max", "expr": "y"}}, time_limit=99999)
    assert r["ok"] is False
    assert r["errors"][0]["where"] == "objective"


def test_report_with_for():
    spec = {
        "sets": {"R": ["n", "s"], "P": ["p1", "p2", "p3"]},
        "params": {"region": {"index": "P", "values": ["n", "n", "s"]}},
        "variables": {"x": {"index": "P", "type": "binary"}},
        "objective": {"sense": "max", "expr": "sum(x[p] for p in P)"},
        "report": {"per_region": {"for": "r in R", "expr": "sum(x[p] for p in P if region[p] == r)"}},
    }
    r = solve(spec)
    assert r["report"]["per_region"] == {"n": 2, "s": 1}
