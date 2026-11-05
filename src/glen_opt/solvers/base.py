"""Interface for solver backends. TODO: define the model format backends receive."""


class SolverBackend:
    name: str = ""

    def available(self) -> bool:
        return True

    def solve(self, model) -> dict:
        raise NotImplementedError
