"""
The path through the model as parametrized by ~ModuleParameters.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch

from rebasin.initialization._permutation import ModuleParameters, Perm


def merge_linear_path(path: Sequence[ModuleParameters]) -> None:
    if len(path) < 2:
        return

    for mod0, mod1 in itertools.pairwise(path):
        # for mypy:
        assert isinstance(mod0, ModuleParameters)
        assert isinstance(mod1, ModuleParameters)

        if len(mod0.output_permutation) == len(mod1.input_permutation):
            mod1.input_permutation = mod0.output_permutation


class LinearPath:
    """A linear path through the model as parametrized by ~ModuleParameters."""
    def __init__(self, path: Sequence[ModuleParameters]) -> None:
        super().__init__()
        self.path = path

    def merge_internal(self) -> None:
        """Merge the internal permutations in the path."""
        merge_linear_path(self.path)

    def apply_permutations(self) -> None:
        for mod in self.path:
            mod.apply_permutations()


class ResidualPath:
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

    def merge_internal(self) -> None:
        """Merge the internal permutations in the path."""
        merge_linear_path(self.long_path)
        merge_linear_path(self.short_path)

    def apply_permutations(self) -> None:
        """Apply the permutations in the short and long path."""
        for mod in self.short_path:
            mod.apply_permutations()
        for mod in self.long_path:
            mod.apply_permutations()


class Path:
    def __init__(self, path: Sequence[LinearPath | ResidualPath]) -> None:
        self.path = path

    def apply_permutations(self) -> None:
        for path in self.path:
            path.apply_permutations()

    def merge_permutations(self) -> None:
        if not self.path:
            return

        for path in self.path:
            self._make_input_identity(path)
            path.merge_internal()
            self._make_output_identity(path)

    def _make_input_identity(self, path: LinearPath | ResidualPath) -> None:
        if isinstance(path, LinearPath):
            self._make_path_input_identity(path.path)
        elif isinstance(path, ResidualPath):
            self._make_path_input_identity(path.long_path)
            if path.short_path:
                self._make_path_input_identity(path.short_path)

    @staticmethod
    def _make_path_input_identity(path: Sequence[ModuleParameters]) -> None:
        """
        Make the input permutation of the first module in the path the identity.

        For one-dimensional weights, the input permutation is also the output
        permutation. Therefore, we need to make
        """
        in_pt = 0  # The input permutation pointer
        identity = Perm(torch.arange(len(path[in_pt].input_permutation)))
        path[in_pt].input_permutation = identity

        while len(path[in_pt].axis_to_permutation) == 1 and in_pt < len(path) - 1:
            in_pt += 1
            path[in_pt].input_permutation = path[in_pt - 1].output_permutation

    def _make_output_identity(self, path: LinearPath | ResidualPath) -> None:
        if isinstance(path, LinearPath):
            self._make_path_output_identity(path.path)
        elif isinstance(path, ResidualPath):
            self._make_path_output_identity(path.long_path)
            if path.short_path:
                self._make_path_output_identity(path.short_path)

    @staticmethod
    def _make_path_output_identity(path: Sequence[ModuleParameters]) -> None:
        """
        Make the output permutation of the last module in the path the identity.

        For one-dimensional weights, the input permutation is also the output
        permutation. Therefore, we need to make
        """
        out_pt = -1  # The output permutation pointer
        identity = Perm(torch.arange(len(path[out_pt].output_permutation)))
        path[out_pt].output_permutation = identity

        while len(path[out_pt].axis_to_permutation) == 1 and -out_pt <= len(path):
            out_pt -= 1
            path[out_pt].output_permutation = path[out_pt + 1].input_permutation
