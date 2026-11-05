"""Build a solver-ready model from a spec (see guide.md for the format)."""

from __future__ import annotations

import ast
import itertools
import json
import keyword
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from . import data
from .expr import (ALLOWED_FUNCS, Affine, Evaluator, NonlinearError, SpecError, Value, canon, evaluate,
                   key_of, parse, parse_for)

MAX_SPEC_BYTES = int(os.environ.get("MAX_SPEC_BYTES", 2_000_000))
MAX_COLS = int(os.environ.get("MAX_COLS", 500_000))
MAX_ROWS = int(os.environ.get("MAX_ROWS", 500_000))
MAX_TERMS = int(os.environ.get("MAX_TERMS", 10_000_000))

VAR_TYPES = ("continuous", "integer", "binary")


class ModelErrors(Exception):
    def __init__(self, errors: list[SpecError]):
        super().__init__(f"{len(errors)} error(s)")
        self.errors = errors


@dataclass
class Param:
    index: list[str]
    values: dict[tuple, Value]
    default: Value = None
    scalar: Value = None


@dataclass
class Var:
    name: str
    index: list[str]
    type: str
    cols: dict[tuple, int]
    description: str = ""
    lb_src: Any = 0
    ub_src: Any = None


@dataclass
class Row:
    name: str
    coefs: dict[int, float]
    lo: float
    hi: float


@dataclass
class Report:
    name: str
    gens: list | None
    loop_names: list[str]
    tree: ast.Expression
    src: str
    description: str


@dataclass
class Model:
    sets: dict[str, list[tuple]] = field(default_factory=dict)
    params: dict[str, Param] = field(default_factory=dict)
    vars: dict[str, Var] = field(default_factory=dict)
    col_names: list[str] = field(default_factory=list)
    col_var: list[tuple[str, tuple]] = field(default_factory=list)
    lb: list[float] = field(default_factory=list)
    ub: list[float] = field(default_factory=list)
    integer: list[bool] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    sense: str | None = None
    objective: Affine = field(default_factory=Affine)
    nonlinear: list[SpecError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reports: list[Report] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)
    terms: int = 0

    @property
    def linear(self) -> bool:
        return not self.nonlinear

    @property
    def model_class(self) -> str:
        has_int = any(self.integer)
        has_cont = not all(self.integer)
        prefix = "MI" if has_int and has_cont else ("I" if has_int else "")
        return prefix + ("LP" if self.linear else "NLP")

    @property
    def nonzeros(self) -> int:
        return sum(len(r.coefs) for r in self.rows)

    def size(self) -> dict:
        return {"variables": len(self.col_names), "integer": sum(self.integer),
                "constraints": len(self.rows), "nonzeros": self.nonzeros}

    # Symbols interface for the evaluator
    def set_members(self, name):
        return self.sets.get(name)

    def scalar(self, name):
        p = self.params.get(name)
        if p is not None and not p.index:
            return p.scalar
        v = self.vars.get(name)
        if v is not None and not v.index:
            return Affine({v.cols[()]: 1.0})
        return None

    def is_indexed(self, name):
        p = self.params.get(name) or self.vars.get(name)
        return p is not None and bool(p.index)

    def indexed(self, name, key, where):
        p = self.params.get(name)
        if p is not None:
            self.check_arity(name, p.index, key, where)
            v = p.values.get(key)
            if v is None:
                if p.default is None:
                    raise SpecError(where, f"'{name}' has no value for {_fmt_key(key)}",
                                    "Add the value or give the param a `default`.")
                return p.default
            return v
        var = self.vars[name]
        self.check_arity(name, var.index, key, where)
        col = var.cols.get(key)
        if col is None:
            raise SpecError(where, f"'{name}' has no element {_fmt_key(key)} (indexed by {', '.join(var.index)})")
        return Affine({col: 1.0})

    def check_arity(self, name, index, key, where):
        width = sum(len(self.sets[s][0]) if self.sets[s] else 1 for s in index)
        if len(key) != width:
            raise SpecError(where, f"'{name}' takes {width} index value(s) ({', '.join(index)}); got {len(key)}")

    def count_term(self):
        self.terms += 1
        if self.terms > MAX_TERMS:
            raise SpecError("model", f"model expands to more than {MAX_TERMS} terms")


def _fmt_key(key: tuple) -> str:
    return "[" + ", ".join(key) + "]"


def load_spec(spec: dict | str) -> dict:
    if isinstance(spec, str):
        if len(spec) > MAX_SPEC_BYTES:
            raise ModelErrors([SpecError("spec", f"spec is larger than {MAX_SPEC_BYTES} bytes")])
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as e:
            raise ModelErrors([SpecError("spec", f"invalid JSON: {e}")]) from None
    if not isinstance(spec, dict):
        raise ModelErrors([SpecError("spec", "spec must be a JSON object")])
    if len(json.dumps(spec, default=str)) > MAX_SPEC_BYTES:
        raise ModelErrors([SpecError("spec", f"spec is larger than {MAX_SPEC_BYTES} bytes")])
    return spec


def build(spec: dict | str, query: Callable[[str], tuple[list, list]] = data.run_query) -> Model:
    spec = load_spec(spec)
    b = _Builder(spec, query)
    b.run()
    if b.errors:
        raise ModelErrors(b.errors)
    return b.m


class _Builder:
    KNOWN = {"spec_version", "description", "sets", "params", "variables", "objective", "constraints", "report"}

    def __init__(self, spec: dict, query):
        self.spec = spec
        self.query = query
        self.m = Model()
        self.errors: list[SpecError] = []
        self.names: set[str] = set()

    def run(self):
        unknown = set(self.spec) - self.KNOWN
        if unknown:
            self.errors.append(SpecError("spec", f"unknown section(s): {', '.join(sorted(unknown))}",
                                         "Sections: " + ", ".join(sorted(self.KNOWN))))
        if self.spec.get("spec_version", 1) != 1:
            self.errors.append(SpecError("spec_version", "only spec_version 1 is supported"))
        for section, fn in (("sets", self.add_set), ("params", self.add_param), ("variables", self.add_var)):
            entries = self.spec.get(section, {})
            if not isinstance(entries, dict):
                self.errors.append(SpecError(section, f"`{section}` must be an object mapping names to definitions"))
                continue
            for name, d in entries.items():
                where = f"{section}.{name}"
                try:
                    self.check_name(name, where)
                    fn(name, d, where)
                except SpecError as e:
                    self.errors.append(e)
        if not self.m.col_names and not self.errors:
            self.errors.append(SpecError("variables", "the model has no variables"))
        if self.errors:
            return  # expressions depend on declarations
        self.add_objective()
        cons = self.spec.get("constraints", {})
        if not isinstance(cons, dict):
            self.errors.append(SpecError("constraints", "`constraints` must be an object mapping names to constraints"))
        else:
            for name, d in cons.items():
                self.add_constraint(name, d)
        rep = self.spec.get("report", {})
        if not isinstance(rep, dict):
            self.errors.append(SpecError("report", "`report` must be an object mapping names to expressions"))
        else:
            for name, d in rep.items():
                self.add_report(name, d)
        self.finish()

    # ---- declarations ----

    def check_name(self, name: str, where: str):
        if not name.isidentifier() or keyword.iskeyword(name) or name in ALLOWED_FUNCS:
            raise SpecError(where, f"'{name}' is not a valid name (use letters, digits, _; not a keyword or {', '.join(sorted(ALLOWED_FUNCS))})")
        if name in self.names:
            raise SpecError(where, f"name '{name}' is already used by another set, param, or variable")
        self.names.add(name)

    def sql(self, d: dict, where: str) -> tuple[list, list]:
        q = d.get("query")
        if not isinstance(q, str):
            raise SpecError(where, "a SQL source needs a `query` string")
        try:
            return self.query(q)
        except ValueError as e:
            raise SpecError(where, str(e)) from None

    def add_set(self, name, d, where):
        if isinstance(d, list):
            raw = d
        elif isinstance(d, dict) and "range" in d:
            n = d["range"]
            if not isinstance(n, int) or n < 0:
                raise SpecError(where, "`range` must be a non-negative integer")
            raw = list(range(n))
        elif isinstance(d, dict) and d.get("source") == "sql":
            _, rows = self.sql(d, where)
            raw = [r[0] if len(r) == 1 else list(r) for r in rows]
        else:
            raise SpecError(where, "a set is a list of labels, {\"range\": n}, or {\"source\": \"sql\", \"query\": ...}")
        members, seen = [], set()
        for x in raw:
            t = tuple(x) if isinstance(x, list) else (x,)
            if any(isinstance(e, (dict, list)) or e is None for e in t):
                raise SpecError(where, f"invalid set element {x!r}")
            k = key_of(t)
            if k in seen:
                raise SpecError(where, f"duplicate element {x!r}")
            seen.add(k)
            members.append(t)
        if members and len({len(t) for t in members}) > 1:
            raise SpecError(where, "all elements must have the same number of components")
        self.m.sets[name] = members

    def index_of(self, d: dict, where: str) -> list[str]:
        idx = d.get("index", [])
        idx = [idx] if isinstance(idx, str) else idx
        if not isinstance(idx, list) or not all(isinstance(s, str) for s in idx):
            raise SpecError(where, "`index` must be a set name or a list of set names")
        for s in idx:
            if s not in self.m.sets:
                raise SpecError(where, f"unknown set '{s}' in index")
        return idx

    def product_keys(self, idx: list[str]) -> list[tuple]:
        return [key_of(sum(parts, ())) for parts in itertools.product(*(self.m.sets[s] for s in idx))]

    def add_param(self, name, d, where):
        if isinstance(d, (int, float)) and not isinstance(d, bool):
            self.m.params[name] = Param([], {}, scalar=d)
            return
        if not isinstance(d, dict):
            raise SpecError(where, "a param is a number, {\"value\": v}, {\"index\": ..., \"values\": ...}, or a SQL source")
        idx = self.index_of(d, where)
        default = d.get("default")
        if not idx:
            if d.get("source") == "sql":
                _, rows = self.sql(d, where)
                if len(rows) != 1 or len(rows[0]) != 1:
                    raise SpecError(where, "a scalar SQL param must return exactly one row with one column")
                v = rows[0][0]
            elif "value" in d:
                v = d["value"]
            else:
                raise SpecError(where, "scalar param needs `value` (or add `index`)")
            self.m.params[name] = Param([], {}, scalar=self.clean_value(v, where))
            return
        widths = [len(self.m.sets[s][0]) if self.m.sets[s] else 1 for s in idx]
        nkey = sum(widths)
        values: dict[tuple, Value] = {}
        if d.get("source") == "sql":
            cols, rows = self.sql(d, where)
            if len(cols) != nkey + 1:
                raise SpecError(where, f"query must return {nkey} index column(s) then 1 value column; got {len(cols)} columns")
            for r in rows:
                values[key_of(r[:-1])] = self.clean_value(r[-1], where)
        elif "values" in d:
            vals = d["values"]
            if isinstance(vals, list):
                if len(idx) != 1:
                    raise SpecError(where, "list `values` only work for a single index; use nested objects")
                members = self.m.sets[idx[0]]
                if len(vals) != len(members):
                    raise SpecError(where, f"`values` has {len(vals)} entries; set '{idx[0]}' has {len(members)}")
                for mem, v in zip(members, vals):
                    values[key_of(mem)] = self.clean_value(v, where)
            elif isinstance(vals, dict):
                self.flatten(vals, len(idx), (), values, where)
            else:
                raise SpecError(where, "`values` must be an object (nested per index) or a list")
        else:
            raise SpecError(where, "indexed param needs `values` or a SQL source")
        valid = set(self.product_keys(idx))
        bad = [k for k in values if k not in valid]
        if bad:
            raise SpecError(where, f"{len(bad)} key(s) not in {' x '.join(idx)}, e.g. {_fmt_key(bad[0])}")
        if default is None:
            missing = [k for k in valid if k not in values]
            if missing:
                raise SpecError(where, f"missing values for {len(missing)} element(s), e.g. {_fmt_key(missing[0])}",
                                "Add them or set a `default`.")
        else:
            default = self.clean_value(default, where)
        self.m.params[name] = Param(idx, values, default=default)

    def flatten(self, vals, depth, prefix, out, where):
        for k, v in vals.items():
            key = prefix + (canon(k),)
            if depth > 1:
                if not isinstance(v, dict):
                    raise SpecError(where, f"expected nested object under {_fmt_key(key)}")
                self.flatten(v, depth - 1, key, out, where)
            else:
                out[key] = self.clean_value(v, where)

    @staticmethod
    def clean_value(v, where):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            if isinstance(v, float) and not math.isfinite(v):
                raise SpecError(where, "values must be finite numbers")
            return v
        if isinstance(v, str):
            return v
        if v is None:
            raise SpecError(where, "null value (SQL NULL?); filter it out or use `default`")
        try:
            return float(v)  # Decimal etc. from SQL
        except (TypeError, ValueError):
            raise SpecError(where, f"unsupported value {v!r}") from None

    def add_var(self, name, d, where):
        if not isinstance(d, dict):
            raise SpecError(where, "a variable is an object: {\"index\": ..., \"type\": ..., \"lb\": ..., \"ub\": ...}")
        unknown = set(d) - {"index", "type", "lb", "ub", "description"}
        if unknown:
            raise SpecError(where, f"unknown field(s): {', '.join(sorted(unknown))}")
        idx = self.index_of(d, where)
        vtype = d.get("type", "continuous")
        if vtype not in VAR_TYPES:
            raise SpecError(where, f"type must be one of {', '.join(VAR_TYPES)}")
        lb_src, ub_src = d.get("lb", 0), d.get("ub")
        keys = self.product_keys(idx) if idx else [()]
        if len(self.m.col_names) + len(keys) > MAX_COLS:
            raise SpecError(where, f"model exceeds {MAX_COLS} variables")
        var = Var(name, idx, vtype, {}, d.get("description", ""), lb_src, ub_src)
        for k in keys:
            lb = self.bound(lb_src, idx, k, -math.inf, where, "lb")
            ub = self.bound(ub_src, idx, k, math.inf, where, "ub")
            if vtype == "binary":
                lb, ub = max(lb, 0.0), min(ub, 1.0)
            if lb > ub:
                raise SpecError(where, f"lb > ub for {name}{_fmt_key(k) if k else ''}")
            var.cols[k] = len(self.m.col_names)
            self.m.col_names.append(name + (_fmt_key(k) if k else ""))
            self.m.col_var.append((name, k))
            self.m.lb.append(lb)
            self.m.ub.append(ub)
            self.m.integer.append(vtype != "continuous")
        self.m.vars[name] = var

    def bound(self, src, idx, key, none_value, where, field_name) -> float:
        if src is None:
            return none_value
        if isinstance(src, (int, float)) and not isinstance(src, bool):
            return float(src)
        if isinstance(src, str) and src in self.m.params:
            p = self.m.params[src]
            if not p.index:
                v = p.scalar
            elif p.index == idx:
                v = p.values.get(key, p.default)
            else:
                raise SpecError(where, f"{field_name} param '{src}' must be scalar or indexed by {idx}")
            if not isinstance(v, (int, float)):
                raise SpecError(where, f"{field_name} param '{src}' has a non-numeric value for {_fmt_key(key)}")
            return float(v)
        raise SpecError(where, f"`{field_name}` must be a number, null, or a param name")

    # ---- expressions ----

    def add_objective(self):
        obj = self.spec.get("objective")
        if obj is None:
            self.m.summary.append("find any feasible solution (no objective)")
            return
        where = "objective"
        if not isinstance(obj, dict) or obj.get("sense") not in ("min", "max") or "expr" not in obj:
            self.errors.append(SpecError(where, "objective needs `sense` (\"min\" or \"max\") and `expr`"))
            return
        self.m.sense = obj["sense"]
        try:
            v = evaluate(parse(obj["expr"], where), self.m, obj["expr"], where)
            if isinstance(v, (str, tuple)):
                raise SpecError(where, "objective must be numeric")
            self.m.objective = v if isinstance(v, Affine) else Affine({}, float(v))
        except NonlinearError as e:
            self.m.nonlinear.append(e)
        except SpecError as e:
            self.errors.append(e)
        desc = obj.get("description")
        self.m.summary.append(f"{'maximize' if obj['sense'] == 'max' else 'minimize'}"
                              f"{' ' + desc if desc else ''}: {obj['expr']}")

    def loop(self, d: dict, where: str):
        f = d.get("for")
        if f is None:
            return None, []
        if not isinstance(f, str):
            raise SpecError(where, "`for` must be a string like \"r in R\"")
        gens = parse_for(f, where + ".for")
        names = []
        for g in gens:
            t = g.target
            names += [t.id] if isinstance(t, ast.Name) else [e.id for e in getattr(t, "elts", []) if isinstance(e, ast.Name)]
        return gens, names

    def instances(self, gens, where):
        if gens is None:
            yield {}
            return
        yield from Evaluator(self.m, "", where).iterate(gens, {})

    def add_constraint(self, name, d):
        where = f"constraints.{name}"
        if isinstance(d, str):
            d = {"expr": d}
        if not isinstance(d, dict) or "expr" not in d:
            self.errors.append(SpecError(where, "a constraint is {\"expr\": ..., \"for\"?: ..., \"description\"?: ...}"))
            return
        src = d["expr"]
        try:
            tree = parse(src, where)
            cmp = tree.body
            if not isinstance(cmp, ast.Compare):
                raise SpecError(where, "constraint needs a comparison (<=, >=, or ==)")
            ops = [type(o) for o in cmp.ops]
            if ops not in ([ast.LtE], [ast.GtE], [ast.Eq], [ast.LtE, ast.LtE], [ast.GtE, ast.GtE]):
                raise SpecError(where, "use one of: a <= b, a >= b, a == b, lo <= e <= hi",
                                "Strict < and > are not supported.")
            gens, names = self.loop(d, where)
            n_before, trivial = len(self.m.rows), 0
            for env in self.instances(gens, where):
                inst = name + ("[" + ", ".join(canon(env[n]) for n in names) + "]" if names else "")
                iw = f"constraints.{inst}"
                parts = [self.numeric(evaluate(ast.Expression(p), self.m, src, iw, env), iw, src)
                         for p in [cmp.left, *cmp.comparators]]
                if len(parts) == 2:
                    diff = _sub(parts[0], parts[1])
                    lo, hi = {ast.LtE: (-math.inf, 0.0), ast.GtE: (0.0, math.inf), ast.Eq: (0.0, 0.0)}[ops[0]]
                else:
                    a, e, c = parts
                    if not (a.is_constant() and c.is_constant()):
                        raise SpecError(iw, "in lo <= e <= hi, lo and hi must not contain variables")
                    lo, hi = (a.const, c.const) if ops[0] is ast.LtE else (c.const, a.const)
                    diff = e
                if diff.is_constant():
                    if not (lo - 1e-9 <= diff.const <= hi + 1e-9):
                        raise SpecError(iw, "constraint can never hold (it contains no variables and is false)")
                    trivial += 1
                    continue
                coefs = {k: v for k, v in diff.coefs.items() if v != 0.0}
                self.m.rows.append(Row(inst, coefs, lo - diff.const, hi - diff.const))
                if len(self.m.rows) > MAX_ROWS:
                    raise SpecError(where, f"model exceeds {MAX_ROWS} constraints")
            if trivial:
                self.m.warnings.append(f"{where}: {trivial} instance(s) contain no variables and were dropped")
            count = len(self.m.rows) - n_before
            desc = d.get("description")
            self.m.summary.append(f"{name}{' (for ' + d['for'] + ')' if gens else ''}"
                                  f"{' — ' + desc if desc else ''}: {src}  [{count} row(s)]")
        except NonlinearError as e:
            self.m.nonlinear.append(e)
        except SpecError as e:
            self.errors.append(e)

    def numeric(self, v, where, src) -> Affine:
        if isinstance(v, (str, tuple)):
            raise SpecError(where, f"expected a numeric expression, got label {v!r}: {src}")
        return v if isinstance(v, Affine) else Affine({}, float(v))

    def add_report(self, name, d):
        where = f"report.{name}"
        if isinstance(d, str):
            d = {"expr": d}
        try:
            if not isinstance(d, dict) or "expr" not in d:
                raise SpecError(where, "a report is an expression string or {\"expr\": ..., \"for\"?: ...}")
            tree = parse(d["expr"], where)
            gens, names = self.loop(d, where)
            for env in self.instances(gens, where):  # validate now, evaluate after solving
                evaluate(tree, self.m, d["expr"], where, env)
            self.m.reports.append(Report(name, gens, names, tree, d["expr"], d.get("description", "")))
        except NonlinearError as e:
            self.m.nonlinear.append(e)
        except SpecError as e:
            self.errors.append(e)

    def finish(self):
        used = set(self.m.objective.coefs)
        for r in self.m.rows:
            used.update(r.coefs)
        unused: dict[str, int] = {}
        for col, (vname, _) in enumerate(self.m.col_var):
            if col not in used:
                unused[vname] = unused.get(vname, 0) + 1
        for vname, n in unused.items():
            total = len(self.m.vars[vname].cols)
            self.m.warnings.append(f"variables.{vname}: {n} of {total} element(s) appear in no constraint or objective")
        for v in self.m.vars.values():
            lb = "-inf" if v.lb_src is None else v.lb_src
            ub = "inf" if v.ub_src is None else v.ub_src
            bounds = "" if v.type == "binary" else f" in [{lb}, {ub}]"
            self.m.summary.append(f"variable {v.name}{'[' + ', '.join(v.index) + ']' if v.index else ''}: "
                                  f"{v.type}{bounds}  ({len(v.cols)}){' — ' + v.description if v.description else ''}")


def _sub(a: Affine, b: Affine) -> Affine:
    out = Affine(dict(a.coefs), a.const - b.const)
    for k, v in b.coefs.items():
        out.coefs[k] = out.coefs.get(k, 0.0) - v
    return out


def eval_reports(m: Model, x: list[float]) -> dict:
    out = {}
    for r in m.reports:
        def value(env):
            v = evaluate(r.tree, m, r.src, f"report.{r.name}", env)
            if isinstance(v, Affine):
                return v.const + sum(c * x[col] for col, c in v.coefs.items())
            return v
        if r.gens is None:
            out[r.name] = _round(value({}))
        else:
                out[r.name] = {", ".join(canon(env[n]) for n in r.loop_names): _round(value(env))
                           for env in Evaluator(m, "", r.name).iterate(r.gens, {})}
    return out


def _round(v):
    if isinstance(v, float):
        r = round(v)
        return int(r) if abs(v - r) < 1e-6 else round(v, 6)
    return v
