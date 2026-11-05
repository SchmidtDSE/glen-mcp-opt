# glen-mcp-opt

An MCP server that lets an LLM turn optimization questions into models and solve them. The LLM writes the model as JSON; the server validates it, classifies it (LP, ILP, MILP, ...), solves it with [HiGHS](https://highs.dev), and returns results in a form the LLM can explain. Built for [GLEN](https://github.com/boettiger-lab/glen); works with any MCP client.

## Tools

| Tool | Purpose |
|---|---|
| `modeling_guide` | Spec format, expression syntax, SQL data sources, modeling patterns, how to explain results |
| `check_model` | Validate a model: errors to fix, or class, size, solver support, and a readable summary |
| `solve_model` | Solve: status, objective, variable values, binding constraints, duals (LP), gap (on timeout), conflicting constraints (infeasible) |
| `list_solvers` | Available solvers and the model classes they handle |

Example spec (choose projects within a budget):

```json
{
  "sets": { "A": ["a1", "a2", "a3"] },
  "params": {
    "cost":    { "index": "A", "values": { "a1": 3, "a2": 5, "a3": 2 } },
    "benefit": { "index": "A", "values": { "a1": 4, "a2": 7, "a3": 2 } },
    "budget": 6
  },
  "variables":   { "x": { "index": "A", "type": "binary" } },
  "objective":   { "sense": "max", "expr": "sum(benefit[a] * x[a] for a in A)" },
  "constraints": { "budget": "sum(cost[a] * x[a] for a in A) <= budget" }
}
```

Expressions are parsed against a whitelist; no code is executed. Sets and params can come from a DuckDB query over S3 Parquet (`{"source": "sql", "query": ...}`), so data rows never pass through the LLM. Full reference: [src/glen_opt/guide.md](src/glen_opt/guide.md).

## Run

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/glen-mcp-opt        # http://localhost:8000/mcp
.venv/bin/pytest
```

Try it from Claude Code:

```bash
claude mcp add --transport http glen-opt http://localhost:8000/mcp
```

Outside the NRP cluster, set `S3_DEFAULT_ENDPOINT=s3-west.nrp-nautilus.io` for SQL sources.

| Variable | Default | |
|---|---|---|
| `PORT` | `8000` | |
| `S3_DEFAULT_ENDPOINT` | `rook-ceph-rgw-nautiluss3.rook` | In-cluster NRP Ceph |
| `MAX_TIME_LIMIT` | `300` | Seconds per solve |
| `MAX_COLS`, `MAX_ROWS` | `500000` | Model size limits |
| `MAX_SQL_ROWS` | `500000` | Rows per SQL source |

## Use with GLEN

Add the server after duckdb-mcp in the app's `layers-input.json`:

```json
"mcp_servers": [
  { "url": "https://duckdb-mcp.nrp-nautilus.io/mcp" },
  { "url": "https://glen-opt.nrp-nautilus.io/mcp", "optional": true }
]
```

## Deploy

Pushes to `main` build `ghcr.io/schmidtdse/glen-mcp-opt:main`.

```bash
kubectl apply -f k8s/
```

## Evaluate

[eval/problems.json](eval/problems.json) holds plain-language questions with known optimal values. The runner sends each to an LLM through open-llm-proxy, executes its tool calls against this server, and scores the result:

```bash
PROXY_KEY=... .venv/bin/python eval/run_eval.py --models kimi qwen3
```

## License

BSD-3-Clause
