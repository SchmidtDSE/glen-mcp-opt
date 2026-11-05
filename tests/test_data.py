import pytest

from glen_opt.data import run_query
from glen_opt.server import solve


def test_select_values():
    cols, rows = run_query("SELECT * FROM (VALUES ('a', 3), ('b', 5)) t(id, cost)")
    assert cols == ["id", "cost"]
    assert rows == [("a", 3), ("b", 5)]


@pytest.mark.parametrize("sql, fragment", [
    ("SELECT * FROM read_csv('/etc/passwd')", "disabled by configuration"),
    ("SELECT 1; SELECT 2", "exactly one SELECT"),
    ("SET threads=1", "exactly one SELECT"),
    ("COPY (SELECT 1) TO 'x.csv'", "exactly one SELECT"),
    ("SELECT nope FROM", "SQL error"),
])
def test_rejected(sql, fragment):
    with pytest.raises(ValueError, match=fragment):
        run_query(sql)


def test_spec_with_sql_sources():
    data = "(VALUES ('a', 3, 4), ('b', 5, 7), ('c', 2, 2)) t(id, cost, benefit)"
    spec = {
        "sets": {"A": {"source": "sql", "query": f"SELECT id FROM {data}"}},
        "params": {
            "cost": {"index": "A", "source": "sql", "query": f"SELECT id, cost FROM {data}"},
            "benefit": {"index": "A", "source": "sql", "query": f"SELECT id, benefit FROM {data}"},
            "budget": {"source": "sql", "query": "SELECT 7"},
        },
        "variables": {"x": {"index": "A", "type": "binary"}},
        "objective": {"sense": "max", "expr": "sum(benefit[a] * x[a] for a in A)"},
        "constraints": {"budget": "sum(cost[a] * x[a] for a in A) <= budget"},
    }
    r = solve(spec)
    assert r["objective"] == 9  # b + c
