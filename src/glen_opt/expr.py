"""Expression language: whitelisted Python syntax, evaluated to linear forms.

Expressions are parsed with `ast` and walked by `Evaluator`; nothing is passed
to eval/exec. A value is a number, a label (str), or an `Affine` (linear form
over model columns). Operations that would produce a nonlinear form raise
`NonlinearError`.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


class SpecError(Exception):
    def __init__(self, where: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.where = where
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict:
        d = {"where": self.where, "message": self.message}
        if self.hint:
            d["hint"] = self.hint
        return d


class NonlinearError(SpecError):
    pass


@dataclass
class Affine:
    """sum(coefs[col] * x[col]) + const"""

    coefs: dict[int, float] = field(default_factory=dict)
    const: float = 0.0

    def is_constant(self) -> bool:
        return not self.coefs


Value = Any  # float | int | bool | str | tuple | Affine

NUMERIC_FUNCS: dict[str, Callable] = {
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
}
REWRITE_HINTS = {
    "abs": "Linearize: add a variable t with t >= e and t >= -e, then use t (see modeling_guide).",
    "min": "Linearize: add a variable t with t <= each argument when maximizing t (see modeling_guide).",
    "max": "Linearize: add a variable t with t >= each argument when minimizing t (see modeling_guide).",
}
ALLOWED_FUNCS = {"sum", "len", "abs", "min", "max", *NUMERIC_FUNCS}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Name,
    ast.Constant, ast.Subscript, ast.Tuple, ast.GeneratorExp, ast.comprehension, ast.IfExp,
    ast.Load, ast.Store, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub,
    ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn,
)


def parse(src: str, where: str) -> ast.Expression:
    if not isinstance(src, str) or not src.strip():
        raise SpecError(where, "expression must be a non-empty string")
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e:
        raise SpecError(where, f"syntax error at column {e.offset}: {e.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SpecError(where, f"unsupported syntax: {type(node).__name__}",
                            "Allowed: arithmetic, indexing, sum(... for i in S if ...), comparisons, "
                            + ", ".join(sorted(ALLOWED_FUNCS)))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                raise SpecError(where, f"unsupported function: {ast.unparse(node.func)}",
                                "Allowed functions: " + ", ".join(sorted(ALLOWED_FUNCS)))
            if node.keywords:
                raise SpecError(where, f"keyword arguments are not supported: {ast.unparse(node)}")
    return tree


def parse_for(src: str, where: str) -> list[ast.comprehension]:
    """Parse a constraint's `for` clause, e.g. "r in R" or "i, j in E if w[i, j] > 0"."""
    tree = parse(f"(0 for {src})", where)
    gen = tree.body
    if not isinstance(gen, ast.GeneratorExp):
        raise SpecError(where, "`for` must look like \"r in R\" or \"r in R for t in T if ...\"")
    return gen.generators


def canon(label: Value) -> str:
    if isinstance(label, bool):
        return str(int(label))
    if isinstance(label, float) and label.is_integer():
        return str(int(label))
    return str(label)


def key_of(labels: tuple) -> tuple[str, ...]:
    return tuple(canon(x) for x in labels)


# ---- arithmetic on Value -------------------------------------------------

def _num(v: Value, where: str, node_src: str) -> float:
    if isinstance(v, Affine):
        if v.is_constant():
            return v.const
        raise NonlinearError(where, f"nonlinear term: {node_src}")
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    raise SpecError(where, f"expected a number, got label {v!r} in: {node_src}")


def _check_arith(v: Value, where: str, node_src: str) -> None:
    if isinstance(v, (str, tuple)):
        raise SpecError(where, f"arithmetic on label {v!r} in: {node_src}",
                        "Labels can be compared (==, !=, in) but not added or multiplied.")


def add(a: Value, b: Value, sign: float = 1.0) -> Value:
    if not isinstance(a, Affine) and not isinstance(b, Affine):
        return a + sign * b
    out = Affine(dict(a.coefs), a.const) if isinstance(a, Affine) else Affine({}, float(a))
    iadd(out, b, sign)
    return out


def iadd(acc: Affine, v: Value, sign: float = 1.0) -> None:
    if isinstance(v, Affine):
        c = acc.coefs
        for col, k in v.coefs.items():
            c[col] = c.get(col, 0.0) + sign * k
        acc.const += sign * v.const
    else:
        acc.const += sign * v


def scale(a: Value, k: float) -> Value:
    if isinstance(a, Affine):
        return Affine({col: k * c for col, c in a.coefs.items()}, k * a.const)
    return a * k


def mul(a: Value, b: Value, where: str, node_src: str) -> Value:
    if isinstance(a, Affine) and isinstance(b, Affine):
        if a.is_constant():
            return scale(b, a.const)
        if b.is_constant():
            return scale(a, b.const)
        raise NonlinearError(where, f"nonlinear term (product of variables): {node_src}",
                             "Products of two binaries, or a binary and a bounded variable, can be "
                             "linearized exactly (see modeling_guide).")
    if isinstance(a, Affine):
        return scale(a, b)
    if isinstance(b, Affine):
        return scale(b, a)
    return a * b


# ---- evaluator -----------------------------------------------------------

class Symbols:
    """Interface the evaluator needs from the model (implemented in model.py)."""

    def set_members(self, name: str) -> list[tuple] | None: ...
    def scalar(self, name: str) -> Value | None: ...
    def indexed(self, name: str, key: tuple[str, ...], where: str) -> Value | None: ...
    def is_indexed(self, name: str) -> bool: ...
    def count_term(self) -> None: ...


class Evaluator:
    def __init__(self, symbols: Symbols, src: str, where: str):
        self.sym = symbols
        self.src = src
        self.where = where

    def seg(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.src, node) or ast.unparse(node)

    def err(self, node: ast.AST, msg: str, hint: str | None = None) -> SpecError:
        return SpecError(self.where, f"{msg}: {self.seg(node)}", hint)

    # iteration over generator clauses
    def iterate(self, gens: list[ast.comprehension], env: dict) -> Iterator[dict]:
        if not gens:
            yield env
            return
        g, rest = gens[0], gens[1:]
        if not isinstance(g.iter, ast.Name):
            raise self.err(g.iter, "can only iterate over a set name")
        members = self.sym.set_members(g.iter.id)
        if members is None:
            raise self.err(g.iter, f"unknown set '{g.iter.id}'")
        if isinstance(g.target, ast.Name):
            names = [g.target.id]
        elif isinstance(g.target, ast.Tuple) and all(isinstance(e, ast.Name) for e in g.target.elts):
            names = [e.id for e in g.target.elts]
        else:
            raise self.err(g.target, "loop target must be a name or a tuple of names")
        for n in names:
            if n in env:
                continue
            if self.sym.set_members(n) is not None or self.sym.scalar(n) is not None or self.sym.is_indexed(n):
                raise self.err(g.target, f"loop variable '{n}' has the same name as a model symbol")
        for m in members:
            if len(names) == 1:
                vals = (m[0] if len(m) == 1 else m,)
            elif len(names) == len(m):
                vals = m
            else:
                raise self.err(g.target, f"set '{g.iter.id}' has {len(m)}-tuples; loop target has {len(names)} names")
            e = {**env, **dict(zip(names, vals))}
            if all(self.truth(self.ev(c, e), c) for c in g.ifs):
                yield from self.iterate(rest, e)

    def truth(self, v: Value, node: ast.AST) -> bool:
        if isinstance(v, Affine):
            if v.is_constant():
                return bool(v.const)
            raise self.err(node, "conditions cannot depend on variables")
        return bool(v)

    def ev(self, node: ast.AST, env: dict) -> Value:
        m = getattr(self, "_" + type(node).__name__, None)
        if m is None:
            raise self.err(node, f"unsupported syntax {type(node).__name__}")
        return m(node, env)

    def _Expression(self, node, env):
        return self.ev(node.body, env)

    def _Constant(self, node, env):
        if isinstance(node.value, (int, float, str, bool)):
            return node.value
        raise self.err(node, "unsupported constant")

    def _Tuple(self, node, env):
        return tuple(self.ev(e, env) for e in node.elts)

    def _Name(self, node, env):
        n = node.id
        if n in env:
            return env[n]
        v = self.sym.scalar(n)
        if v is not None:
            return v
        if self.sym.is_indexed(n):
            raise self.err(node, f"'{n}' is indexed; use {n}[...]")
        if self.sym.set_members(n) is not None:
            raise self.err(node, f"'{n}' is a set; use it in `for ... in {n}` or len({n})")
        raise self.err(node, f"unknown name '{n}'")

    def _Subscript(self, node, env):
        if not isinstance(node.value, ast.Name):
            raise self.err(node, "only names can be indexed")
        name = node.value.id
        if not self.sym.is_indexed(name):
            if self.sym.scalar(name) is None and self.sym.set_members(name) is None and name not in env:
                raise self.err(node, f"unknown name '{name}'")
            raise self.err(node, f"'{name}' is not an indexed parameter or variable")
        idx = node.slice
        parts = idx.elts if isinstance(idx, ast.Tuple) else [idx]
        labels = []
        for p in parts:
            v = self.ev(p, env)
            if isinstance(v, Affine):
                raise self.err(p, "an index cannot depend on variables")
            labels.extend(v if isinstance(v, tuple) else (v,))
        return self.sym.indexed(name, key_of(tuple(labels)), self.where)

    def _UnaryOp(self, node, env):
        v = self.ev(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not self.truth(v, node.operand)
        _check_arith(v, self.where, self.seg(node))
        return scale(v, -1.0) if isinstance(node.op, ast.USub) else v

    def _BinOp(self, node, env):
        a, b = self.ev(node.left, env), self.ev(node.right, env)
        s = self.seg(node)
        _check_arith(a, self.where, s)
        _check_arith(b, self.where, s)
        op = node.op
        if isinstance(op, ast.Add):
            return add(a, b)
        if isinstance(op, ast.Sub):
            return add(a, b, -1.0)
        if isinstance(op, ast.Mult):
            return mul(a, b, self.where, s)
        if isinstance(op, ast.Div):
            d = _num(b, self.where, s)
            if d == 0:
                raise self.err(node, "division by zero")
            return scale(a, 1.0 / d)
        if isinstance(op, ast.Pow):
            e = _num(b, self.where, s)
            if isinstance(a, Affine) and not a.is_constant():
                if e == 1:
                    return a
                if e == 0:
                    return 1.0
                raise NonlinearError(self.where, f"nonlinear term: {s}",
                                     "Powers of variables are nonlinear; consider a piecewise-linear "
                                     "approximation (see modeling_guide).")
            return _num(a, self.where, s) ** e
        if isinstance(op, ast.Mod):
            return _num(a, self.where, s) % _num(b, self.where, s)
        raise self.err(node, "unsupported operator")

    def _BoolOp(self, node, env):
        if isinstance(node.op, ast.And):
            return all(self.truth(self.ev(v, env), v) for v in node.values)
        return any(self.truth(self.ev(v, env), v) for v in node.values)

    def _IfExp(self, node, env):
        return self.ev(node.body if self.truth(self.ev(node.test, env), node.test) else node.orelse, env)

    def _Compare(self, node, env):
        left = self.ev(node.left, env)
        for op, right_node in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                if not isinstance(right_node, ast.Name) or self.sym.set_members(right_node.id) is None:
                    raise self.err(node, "`in` requires a set name on the right")
                members = {key_of(m) for m in self.sym.set_members(right_node.id)}
                k = key_of(left if isinstance(left, tuple) else (left,))
                ok = (k in members) == isinstance(op, ast.In)
                right = None
            else:
                right = self.ev(right_node, env)
                if isinstance(left, Affine) or isinstance(right, Affine):
                    raise self.err(node, "comparisons inside expressions cannot depend on variables",
                                   "Write constraints as their own entries under `constraints`.")
                ok = self._cmp(op, left, right)
            if not ok:
                return False
            left = right
        return True

    @staticmethod
    def _cmp(op, a, b) -> bool:
        if isinstance(a, str) != isinstance(b, str):
            a, b = canon(a), canon(b)
        if isinstance(op, ast.Eq):
            return a == b
        if isinstance(op, ast.NotEq):
            return a != b
        if isinstance(op, ast.Lt):
            return a < b
        if isinstance(op, ast.LtE):
            return a <= b
        if isinstance(op, ast.Gt):
            return a > b
        return a >= b

    def _Call(self, node, env):
        fn = node.func.id
        if fn == "sum":
            if len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
                raise self.err(node, "sum takes one generator: sum(expr for i in S)")
            gen = node.args[0]
            acc = Affine()
            for e in self.iterate(gen.generators, env):
                self.sym.count_term()
                v = self.ev(gen.elt, e)
                _check_arith(v, self.where, self.seg(gen.elt))
                iadd(acc, v)
            return acc if acc.coefs else acc.const
        if fn == "len":
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Name) \
                    or self.sym.set_members(node.args[0].id) is None:
                raise self.err(node, "len takes a set name")
            return len(self.sym.set_members(node.args[0].id))
        if len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
            gen = node.args[0]
            args = [self.ev(gen.elt, e) for e in self.iterate(gen.generators, env)]
        else:
            args = [self.ev(a, env) for a in node.args]
        if any(isinstance(a, Affine) and not a.is_constant() for a in args):
            raise NonlinearError(self.where, f"nonlinear function of variables: {self.seg(node)}",
                                 REWRITE_HINTS.get(fn, "Consider a piecewise-linear approximation (see modeling_guide)."))
        nums = [_num(a, self.where, self.seg(node)) for a in args]
        try:
            if fn == "abs":
                return abs(nums[0])
            if fn in ("min", "max"):
                return (min if fn == "min" else max)(nums)
            return NUMERIC_FUNCS[fn](*nums)
        except (ValueError, TypeError, IndexError) as e:
            raise self.err(node, f"cannot evaluate ({e})") from None


def evaluate(tree: ast.Expression, symbols: Symbols, src: str, where: str, env: dict | None = None) -> Value:
    return Evaluator(symbols, src, where).ev(tree, env or {})
