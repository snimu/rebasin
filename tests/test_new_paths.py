from __future__ import annotations

import copy

import torch
from torch import nn

from rebasin.modules import (  # type: ignore[attr-defined]
    Permutation,
    initialize_module,
)
from rebasin.paths import LinearPath
from tests.fixtures.utils import allclose, model_change_percent


class TestLinearPath:
    @property
    def lin_path(self) -> tuple[nn.Module, LinearPath]:
        ln1 = nn.LayerNorm(10)
        lin1 = nn.Linear(10, 10)
        lin2 = nn.Linear(10, 10)
        ln2 = nn.LayerNorm(10)

        model = nn.Sequential(ln1, lin1, nn.ReLU(), lin2, nn.ReLU(), ln2)

        mod0 = initialize_module(ln1, ln1)
        mod1 = initialize_module(lin1, lin1)
        mod2 = initialize_module(lin2, lin2)
        mod3 = initialize_module(ln2, ln2)

        path = LinearPath(mod0, mod1, mod2, mod3)
        return model, path

    @property
    def conv_path(self) -> tuple[nn.Module, LinearPath]:
        ln1 = nn.LayerNorm([3, 10, 10])
        conv1 = nn.Conv2d(3, 3, (3, 3))
        conv2 = nn.Conv2d(3, 3, (3, 3))
        ln2 = nn.LayerNorm([3, 6, 6])

        model = nn.Sequential(ln1, conv1, nn.ReLU(), conv2, nn.ReLU(), ln2)

        mod0 = initialize_module(ln1, ln1)
        mod1 = initialize_module(conv1, conv1)
        mod2 = initialize_module(conv2, conv2)
        mod3 = initialize_module(ln2, ln2)

        path = LinearPath(mod0, mod1, mod2, mod3)
        return model, path

    def test_len(self) -> None:
        _, path = self.lin_path
        assert len(path) == 4

    def test_iter(self) -> None:
        _, path = self.lin_path
        assert list(path) == list(path.modules)

    def test_getitem(self) -> None:
        _, path = self.lin_path
        assert path[0] == path.modules[0]
        assert path[1] == path.modules[1]

    def test_bool(self) -> None:
        _, path = self.lin_path
        assert bool(path) is True
        assert not LinearPath()

    def test_input_permutation(self) -> None:
        _, lin_path = self.lin_path
        assert lin_path.input_permutation == lin_path[0].input_permutation

        permutation = Permutation(torch.randperm(10))
        lin_path.input_permutation = permutation
        assert lin_path.input_permutation is permutation
        assert lin_path[1].input_permutation is permutation  # It's a 1d-module

        _, conv_path = self.conv_path
        assert conv_path.input_permutation == conv_path[0].input_permutation

        permutation = Permutation(torch.randperm(3))
        conv_path.input_permutation = permutation
        assert conv_path.input_permutation is permutation
        assert conv_path[1].input_permutation is permutation  # It's a 2d-module
        assert conv_path[1].output_permutation is not permutation

    def test_output_permutation(self) -> None:
        _, lin_path = self.lin_path
        assert lin_path.output_permutation == lin_path[-1].output_permutation

        permutation = Permutation(torch.randperm(10))
        lin_path.output_permutation = permutation
        assert lin_path.output_permutation is permutation
        assert lin_path[-1].input_permutation is permutation  # It's a 1d-module
        assert lin_path[-2].output_permutation is permutation  # It's 1d and a LayerNorm

        _, conv_path = self.conv_path
        assert conv_path.output_permutation == conv_path[-1].output_permutation

        permutation = Permutation(torch.randperm(3))
        conv_path.output_permutation = permutation
        assert conv_path.output_permutation is permutation
        assert conv_path[-1].output_permutation is permutation  # It's a LayerNorm
        assert conv_path[-1].input_permutation is permutation
        assert conv_path[-2].output_permutation is permutation

    def test_io_linear(self) -> None:
        lin_model, lin_path = self.lin_path
        lin_model_orig = copy.deepcopy(lin_model)
        x = torch.randn(10, 10)
        y_orig = lin_model(x)

        for permutation, _ in lin_path.permutation_to_info:
            permutation.perm_indices = torch.randperm(len(permutation.perm_indices))

        lin_path.input_permutation = None
        lin_path.output_permutation = None

        lin_path.enforce_identity()
        lin_path.apply_permutations()
        assert model_change_percent(lin_model, lin_model_orig) > 0.1

        y_new = lin_model(x)
        assert allclose(y_orig, y_new)

    def test_io_conv(self) -> None:
        conv_model, conv_path = self.conv_path
        conv_model_orig = copy.deepcopy(conv_model)
        x = torch.randn(1, 3, 10, 10)
        y_orig = conv_model(x)

        for permutation, _ in conv_path.permutation_to_info:
            while torch.all(
                    permutation.perm_indices
                    == torch.arange(len(permutation.perm_indices))
            ):
                permutation.perm_indices = torch.randperm(len(permutation.perm_indices))

        conv_path.input_permutation = None
        conv_path.output_permutation = None

        conv_path.enforce_identity()
        conv_path.apply_permutations()
        assert model_change_percent(conv_model, conv_model_orig) > 0.02

        y_new = conv_model(x)
        assert allclose(y_orig, y_new)

    @staticmethod
    def test_empty_path() -> None:
        path = LinearPath()
        assert len(path) == 0
        assert not path
        assert not path.modules
        assert not path.permutation_to_info
        assert path.input_permutation is None
        assert path.output_permutation is None
        path.input_permutation = torch.randperm(4)
        assert path.input_permutation is None
        path.output_permutation = torch.randperm(4)
        assert path.output_permutation is None
        path.enforce_identity()
        path.apply_permutations()

