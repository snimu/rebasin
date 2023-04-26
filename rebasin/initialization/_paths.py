"""
The path through the model as parametrized by ~ModuleParameters.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from rebasin.initialization._permutation import ModuleParameters


class BasePath:

    @staticmethod
    def _merge_linear_path(path: Sequence[ModuleParameters]) -> None:
        if len(path) < 2:
            return

        for mod0, mod1 in itertools.pairwise(path):
            if len(mod0.output_permutation) == len(mod1.input_permutation):
                mod0.output_permutation = mod1.input_permutation


class LinearPath(BasePath):
    """A linear path through the model as parametrized by ~ModuleParameters."""
    def __init__(self, path: Sequence[ModuleParameters]) -> None:
        super().__init__()
        self.path = path
        self._merge_linear_path(self.path)

    def apply_permutations(self) -> None:
        for mod in self.path:
            mod.apply_permutations()


class ResidualPath(BasePath):
    """
    A residual path through the model as parametrized by ~ModuleParameters.
    """
    def __init__(
            self,
            long_path: Sequence[ModuleParameters],
            short_path: Sequence[ModuleParameters]
    ) -> None:
        super().__init__()

        if not long_path:
            raise ValueError("The long path must be non-empty.")

        self.long_path = long_path
        self.short_path = short_path

        self._merge_linear_path(self.long_path)
        self._merge_linear_path(self.short_path)

    def apply_permutations(self) -> None:
        """Apply the permutations in the short and long path."""
        for mod in self.short_path:
            mod.apply_permutations()
        for mod in self.long_path:
            mod.apply_permutations()

