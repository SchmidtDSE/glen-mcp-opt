# glen-mcp-opt

An MCP server that lets [GLEN](https://github.com/boettiger-lab/glen) answer optimization questions.

## Goal

Users ask questions in plain language, e.g. *"Which parcels should I invest in to maximize species connectivity?"* GLEN translates the question into an optimization model, solves it, and explains the answer.

## Flow

1. **Clarify:** identify the decision, objective, and limits, and ask the user for anything missing (e.g. budget).
2. **Get data:** query the relevant datasets (cost, area, habitat, ...).
3. **Formulate:** map the question to a problem template.
4. **Solve:** call a solver.
5. **Explain:** report the answer, which constraints were binding, and the trade-offs.

## Problem templates

Common question types, each with inputs the agent fills in. TODO: decide the list.

- `select_parcels`: choose sites to maximize a benefit within a budget
- TODO
- TODO

## Solvers

TODO: choose open-source solvers.

## Open questions

- What questions will users ask?
- How is connectivity defined, and what data supports it?
- Where do the agent's instructions live? (GLEN doesn't load prompts from secondary MCP servers, so tool descriptions are the main channel.)
- How large are problems expected to be?

## Use with GLEN

Add the server after duckdb-mcp in `layers-input.json`:

```json
"mcp_servers": [
  { "url": "https://duckdb-mcp.nrp-nautilus.io/mcp" },
  { "url": "https://<opt-host>/mcp", "optional": true }
]
```

## Run

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/glen-mcp-opt        # http://localhost:8000/mcp
```
