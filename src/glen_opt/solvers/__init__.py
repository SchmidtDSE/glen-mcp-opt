"""Solver registry."""

from .base import Solution, SolverBackend
from .highs import HighsBackend

_REGISTRY: dict[str, SolverBackend] = {}


def register(backend: SolverBackend) -> None:
    _REGISTRY[backend.name] = backend


def available() -> list[SolverBackend]:
    return [b for b in _REGISTRY.values() if b.available()]


def for_class(model_class: str) -> list[SolverBackend]:
    return [b for b in available() if model_class in b.classes]


def get(name: str) -> SolverBackend | None:
    b = _REGISTRY.get(name)
    return b if b and b.available() else None


register(HighsBackend())

__all__ = ["Solution", "SolverBackend", "register", "available", "for_class", "get"]
