"""
The path through the model as parametrized by ~ModuleParameters.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rebasin import utils
from rebasin.structs import ModuleParameters, Permutation


def merge_linear_path(path: Sequence[ModuleParameters]) -> None:
    if len(path) < 2:
        return

    for mod0, mod1 in utils.pairwise(path):
        # for mypy:
        assert isinstance(mod0, ModuleParameters)
        assert isinstance(mod1, ModuleParameters)

        # Make the ModuleParameter.output_permutation.setter fix any issues.
        # This is important because nD-LayerNorms with n>1 must not have
        #   their output dimension changed.
        mod0.output_permutation = mod0.output_permutation
        mod1.output_permutation = mod1.output_permutation

        # LayerNorms behave differently than other modules
        #   (see test_output_consistency).
        if (
            mod0.module_type is nn.LayerNorm
            and len(mod0.input_permutation) == len(mod1.input_permutation)
        ):
            mod1.input_permutation = mod0.input_permutation
        elif len(mod0.output_permutation) == len(mod1.input_permutation):
            mod1.input_permutation = mod0.output_permutation


class ModelPaths:
    def __init__(
            self,
            paths: Sequence[Sequence[ModuleParameters]],
            enforce_identity: bool = True
    ) -> None:
        self.paths = paths
        self.enforce_identity = enforce_identity

    def apply_permutations(self) -> None:
        # Make sure that the constraints on the permutations are applied:
        #  1. They are merged
        #  2. They change the model in such a way
        #       that it produces the same output as before
        self.merge_permutations()

        # Apply the permutations to the model
        for path in self.paths:
            for mod in path:
                mod.apply_permutations()

    def merge_permutations(self) -> None:
        for path in self.paths:
            if not path:
                continue

            if self.enforce_identity:
                self._make_input_identity(path)

            merge_linear_path(path)

            if self.enforce_identity:
                self._make_output_identity(path)

    @staticmethod
    def _make_input_identity(path: Sequence[ModuleParameters]) -> None:
        """
        Make the input permutation of the first module in the path the identity.

        For one-dimensional weights, the input permutation is also the output
        permutation. Therefore, we need to make
        """
        in_pt = 0  # The input permutation pointer
        identity = Permutation(torch.arange(len(path[in_pt].input_permutation)))
        path[in_pt].input_permutation = identity

        # Handle multi-dim LayerNorm inputs
        path[in_pt].output_permutation = path[in_pt].output_permutation

        # Handle 1d LayerNorm inputs
        while len(path[in_pt].axis_to_permutation) == 1 and in_pt < len(path) - 1:
            in_pt += 1
            path[in_pt].input_permutation = path[in_pt - 1].output_permutation

    @staticmethod
    def _make_output_identity(path: Sequence[ModuleParameters]) -> None:
        """
        Make the output permutation of the last module in the path the identity.

        For one-dimensional weights, the input permutation is also the output
        permutation. Therefore, we need to make
        """
        out_pt = -1  # The output permutation pointer
        identity = Permutation(torch.arange(len(path[out_pt].output_permutation)))
        path[out_pt].output_permutation = identity

        # Handle LayerNorm as well as 1d outputs
        while (
                (
                        path[out_pt].module_type is nn.LayerNorm
                        or len(path[out_pt].axis_to_permutation) == 1
                )
                and -out_pt < len(path)
        ):
            out_pt -= 1
            identity = Permutation(
                torch.arange(len(path[out_pt + 1].input_permutation))
            )
            path[out_pt + 1].input_permutation = identity
            path[out_pt].output_permutation = path[out_pt + 1].input_permutation
