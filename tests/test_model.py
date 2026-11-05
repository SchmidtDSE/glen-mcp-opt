import pytest

from glen_opt.model import ModelErrors, build


def errors(spec):
    with pytest.raises(ModelErrors) as e:
        build(spec)
    return [(x.where, x.message + " " + (x.hint or "")) for x in e.value.errors]


BASE = {
    "sets": {"A": ["a1", "a2", "a3"]},
    "params": {"cost": {"index": "A", "values": {"a1": 3, "a2": 5, "a3": 2}}, "budget": 6},
    "variables": {"x": {"index": "A", "type": "binary"}},
    "objective": {"sense": "max", "expr": "sum(x[a] for a in A)"},
    "constraints": {"budget": "sum(cost[a] * x[a] for a in A) <= budget"},
}


def with_(**parts):
    return {**BASE, **parts}


def test_build_basic():
    m = build(BASE)
    assert m.model_class == "ILP"
    assert m.size() == {"variables": 3, "integer": 3, "constraints": 1, "nonzeros": 3}
    assert m.rows[0].coefs == {0: 3.0, 1: 5.0, 2: 2.0}
    assert m.rows[0].hi == 6


def test_json_string_spec():
    import json
    assert build(json.dumps(BASE)).model_class == "ILP"


def test_classes():
    cont = with_(variables={"x": {"index": "A"}})
    assert build(cont).model_class == "LP"
    mixed = with_(variables={"x": {"index": "A", "type": "binary"}, "y": {}},
                  constraints={"c": "y <= sum(x[a] for a in A)"})
    assert build(mixed).model_class == "MILP"
    nl = with_(objective={"sense": "max", "expr": "sum(x[a] * x[a] for a in A)"})
    m = build(nl)
    assert m.model_class == "INLP"
    assert "product of variables" in m.nonlinear[0].message


def test_indexed_constraint_and_filters():
    spec = {
        "sets": {"P": ["p1", "p2", "p3"], "R": ["n", "s"]},
        "params": {"region": {"index": "P", "values": {"p1": "n", "p2": "n", "p3": "s"}},
                   "cap": {"index": "R", "values": [1, 1]}},
        "variables": {"x": {"index": "P", "type": "binary"}},
        "objective": {"sense": "max", "expr": "sum(x[p] for p in P)"},
        "constraints": {"cap": {"for": "r in R", "expr": "sum(x[p] for p in P if region[p] == r) <= cap[r]"}},
    }
    m = build(spec)
    assert [r.name for r in m.rows] == ["cap[n]", "cap[s]"]
    assert m.rows[0].coefs == {0: 1.0, 1: 1.0}


def test_range_and_tuple_sets():
    spec = {
        "sets": {"I": {"range": 3}, "E": [["a", "b"], ["b", "c"]], "N": ["a", "b", "c"]},
        "variables": {"x": {"index": "I"}, "y": {"index": "N", "type": "binary"}},
        "objective": {"sense": "min", "expr": "x[2]"},
        "constraints": {
            "step": {"for": "i in I if i < 2", "expr": "x[i+1] >= x[i] + 1"},
            "pair": {"for": "i, j in E", "expr": "y[i] + y[j] <= 1"},
        },
    }
    m = build(spec)
    assert [r.name for r in m.rows] == ["step[0]", "step[1]", "pair[a, b]", "pair[b, c]"]


def test_chained_and_equality():
    m = build(with_(constraints={"c": "1 <= sum(x[a] for a in A) <= 2", "d": "x['a1'] == 1"}))
    assert (m.rows[0].lo, m.rows[0].hi) == (1, 2)
    assert (m.rows[1].lo, m.rows[1].hi) == (1, 1)


def test_param_bounds_and_defaults():
    spec = with_(params={"ub": {"index": "A", "values": {"a1": 4}, "default": 1}},
                 variables={"x": {"index": "A", "type": "integer", "ub": "ub"}},
                 constraints={})
    m = build(spec)
    assert m.ub == [4, 1, 1]


@pytest.mark.parametrize("change, fragment", [
    ({"objective": {"sense": "max", "expr": "sum(y[a] for a in A)"}}, "unknown name 'y'"),
    ({"constraints": {"c": "sum(cost[a] for a in A) < 3"}}, "Strict"),
    ({"constraints": {"c": "x['zz'] <= 1"}}, "no element [zz]"),
    ({"constraints": {"c": "cost['a1', 'a2'] * x['a1'] <= 1"}}, "takes 1 index value"),
    ({"constraints": {"c": "sum(a * x[a] for a in A) <= 1"}}, "arithmetic on label"),
    ({"constraints": {"c": "x.__class__ <= 1"}}, "unsupported syntax"),
    ({"constraints": {"c": "__import__('os') <= 1"}}, "unsupported function"),
    ({"constraints": {"c": "cost['a1'] <= 1"}}, "can never hold"),
    ({"params": {"cost": {"index": "A", "values": {"a1": 3}}, "budget": 6}}, "missing values"),
    ({"params": {"cost": {"index": "A", "values": {"a1": 3, "a2": 1, "a3": 1, "a9": 1}}, "budget": 6}}, "not in A"),
    ({"variables": {"x": {"index": "B"}}}, "unknown set 'B'"),
    ({"variables": {"x": {"index": "A", "type": "bool"}}}, "type must be"),
    ({"sets": {"sum": [1]}}, "not a valid name"),
    ({"extra": 1}, "unknown section"),
])
def test_errors(change, fragment):
    msgs = errors(with_(**change))
    assert any(fragment in w + ": " + m for w, m in msgs), msgs


def test_errors_collected_per_constraint():
    msgs = errors(with_(constraints={"c1": "y <= 1", "c2": "z <= 1"}))
    assert {w for w, _ in msgs} == {"constraints.c1", "constraints.c2"}


def test_warnings_unused_and_trivial():
    m = build(with_(constraints={"c": "x['a1'] <= 1", "t": {"for": "a in A", "expr": "cost[a] >= 0"}},
                    objective={"sense": "max", "expr": "x['a1']"}))
    assert any("2 of 3 element(s) appear in no constraint" in w for w in m.warnings)
    assert any("3 instance(s) contain no variables" in w for w in m.warnings)


def test_sql_source_with_injected_query():
    calls = []

    def fake(q):
        calls.append(q)
        if "ids" in q:
            return ["id"], [("a1",), ("a2",)]
        return ["id", "cost"], [("a1", 3), ("a2", 5)]

    spec = {
        "sets": {"A": {"source": "sql", "query": "SELECT id FROM ids"}},
        "params": {"cost": {"index": "A", "source": "sql", "query": "SELECT id, cost FROM t"}},
        "variables": {"x": {"index": "A", "type": "binary"}},
        "objective": {"sense": "min", "expr": "sum(cost[a] * x[a] for a in A)"},
    }
    m = build(spec, query=fake)
    assert m.objective.coefs == {0: 3.0, 1: 5.0}
    assert len(calls) == 2
