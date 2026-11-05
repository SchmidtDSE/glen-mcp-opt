"""Solver registry. TODO: register backends once solvers are chosen."""

from .base import SolverBackend

_REGISTRY: dict[str, SolverBackend] = {}


def register(backend: SolverBackend) -> None:
    _REGISTRY[backend.name] = backend


def available() -> list[SolverBackend]:
    return [b for b in _REGISTRY.values() if b.available()]
