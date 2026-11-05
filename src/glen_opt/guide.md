# Optimization modeling guide

## Workflow

1. Identify the decisions, the objective, and the constraints in the user's question.
2. Ask the user for missing inputs (budget, targets, limits). Do not invent numbers.
3. Get data. For dataset values, use a SQL source (below) instead of copying rows into the spec.
4. Write the spec and call `check_model`. Fix every error and call it again.
5. If `solvable` is false: rewrite exactly (see Patterns) if possible. Otherwise describe an approximation to the user in plain terms (what it simplifies, how the answer may differ) and ask before using it.
6. Call `solve_model`.
7. Explain the result in the user's terms (see Results).

## Spec

```json
{
  "sets":        { "A": ["a1", "a2", "a3"] },
  "params":      { "cost":    { "index": "A", "values": { "a1": 3, "a2": 5, "a3": 2 } },
                   "benefit": { "index": "A", "values": [4, 7, 2] },
                   "budget": 6 },
  "variables":   { "x": { "index": "A", "type": "binary", "description": "fund action a" } },
  "objective":   { "sense": "max", "expr": "sum(benefit[a] * x[a] for a in A)", "description": "total benefit" },
  "constraints": { "budget": { "expr": "sum(cost[a] * x[a] for a in A) <= budget", "description": "stay within budget" } },
  "report":      { "spent": "sum(cost[a] * x[a] for a in A)" }
}
```

| Section | Form |
|---|---|
| `sets` | List of labels `["a", "b"]`; list of tuples `[["a", "b"], ["b", "c"]]` (e.g. edges); `{"range": n}` for 0..n-1 (allows `i+1`); or a SQL source |
| `params` | Number; `{"value": v}`; `{"index": "A" or ["A", "B"], "values": {...}}` with nested objects per index (or a list in set order for one index); or a SQL source. Optional `default` for missing entries. Values may be numbers or labels. |
| `variables` | `index` (optional), `type`: `continuous` (default), `integer`, `binary`; `lb` (default 0), `ub` (default none): number, `null` (unbounded), or a param name; `description` |
| `objective` | `sense`: `min` or `max`; `expr`; `description`. Omit to find any feasible solution. |
| `constraints` | name → `{"expr", "for"?, "description"?}`. `expr` uses `<=`, `>=`, `==`, or `lo <= e <= hi`. `for` repeats the constraint, e.g. `"r in R"`, `"i, j in E"`, `"r in R for t in T if cap[r, t] > 0"`. |
| `report` | name → expression (or `{"expr", "for"}`), evaluated at the solution. Use for totals the user will ask about. |

Variables default to `lb = 0`. Set `"lb": null` for variables that can be negative.

## Expressions

Python syntax, restricted to:

- numbers, labels (`"north"`), names, indexing `x[a]`, `d[i, j]`
- `+ - * / ** %`
- `sum(expr for a in A if cond)`, nested `for`s allowed
- conditions on params and labels: `== != < <= > >=`, `in` / `not in` a set, `and`, `or`, `not`
- `a if cond else b` (cond must not involve variables)
- `len(A)`, and on params only: `abs`, `min`, `max`, `exp`, `log`, `sqrt`, `sin`, `cos`

`x[a] * y[a]`, `x[a]**2`, `abs(x[a])` and functions of variables are nonlinear. HiGHS (the current solver) handles linear models only: LP, ILP, MILP.

## SQL data sources

Sets and params can be filled by a DuckDB query against the same S3 Parquet data the data tools use. Rows go to the solver, not through the conversation.

```json
"sets":   { "P": { "source": "sql", "query": "SELECT parcel_id FROM read_parquet('s3://bucket/parcels.parquet') WHERE county = 'Marin'" } },
"params": { "cost": { "index": "P", "source": "sql", "query": "SELECT parcel_id, cost FROM read_parquet('s3://bucket/parcels.parquet') WHERE county = 'Marin'" } }
```

- Set: one column (labels) or several (tuples).
- Indexed param: index column(s) first, value column last.
- Scalar param: one row, one column.
- One SELECT per source. Filter out NULLs or give the param a `default`.
- Check row counts with the data tools first; keep models under ~100k variables.

## Patterns

Exact rewrites (same answer):

| Situation | Rewrite |
|---|---|
| `x*y`, both binary | New binary `z`; `z <= x`, `z <= y`, `z >= x + y - 1`; use `z` |
| `b*y`, `b` binary, `0 <= y <= U` | New `z`; `z <= U*b`, `z <= y`, `z >= y - U*(1 - b)` |
| minimize `abs(e)` | New `t`; `t >= e`, `t >= -e`; minimize `t` |
| minimize the max of `e[i]` | New `t`; `t >= e[i]` for all i; minimize `t` |
| maximize the min of `e[i]` | New `t`; `t <= e[i]` for all i; maximize `t` |
| fixed cost if used | Binary `open`; `y <= U * open`; cost `F * open` |
| at most k chosen | `sum(x[a] for a in A) <= k` |
| if a then b | `x[a] <= x[b]` |
| either constraint 1 or 2 | Binary `s`; `e1 <= M*s`, `e2 <= M*(1 - s)` with M a valid upper bound |
| neighbors chosen together (pairs set `E`) | Binary `z[i, j]` per pair, product rewrite above |

Approximations (ask the user first):

| Situation | Approximation | What changes |
|---|---|---|
| Integer model too slow | Solve with a time limit; report the gap | May not be the best; the gap bounds how far |
| Integer model too slow | Make variables continuous (LP relaxation) | Fractional decisions; objective is an optimistic bound |
| Nonlinear curve f(y), y in [0, U] | Piecewise-linear: split y into segments `y[k]` with `0 <= y[k] <= width[k]`, `y = sum(y[k])`, f ≈ `sum(slope[k] * y[k])` | Accuracy depends on the number of segments. Exact formulation without binaries only for concave f when maximizing (convex when minimizing). |
| Too many units | Aggregate units (e.g. coarser regions) | Less spatial detail |

Multiple objectives: optimize one and cap the others as constraints (ε-constraint); re-solve with different caps to show the trade-off. Or combine with weights the user agrees on.

## Results

| `status` | Meaning |
|---|---|
| `optimal` | No better solution exists |
| `time_limit` | Best found so far; `gap` = how far from the best possible (0.02 = within 2%) |
| `infeasible` | Constraints conflict; `conflicting_constraints` lists a conflicting set when available |
| `unbounded` | A constraint or bound is missing |
| `unsupported` | No available solver handles the model class |

- `binding_constraints`: limits that hold the solution back. For LPs, `dual` = change in objective per unit increase of the constraint's right-hand side.
- Translate for non-experts: "The budget is the limiting factor; each extra $1,000 adds about 12 acres" rather than "the budget constraint is binding with dual 0.012".
- State quality: "optimal", or "within 1.5% of the best possible".
- For infeasible models, say which requirements cannot all be met and ask which to relax.
