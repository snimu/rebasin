from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from rebasin.modules import (
    DefaultModule,
    ModuleBase,
    MultiheadAttentionModule,
    Permutation,
    PermutationInfo,
    initialize_module,
)
from tests.fixtures.utils import model_change_percent


class TestPermutation:
    @staticmethod
    def test_len() -> None:
        permutation = Permutation(torch.randperm(5))
        assert len(permutation) == len(permutation.perm_indices) == 5

    @staticmethod
    def test_eq() -> None:
        permutation = Permutation(torch.tensor([0, 2, 1, 3]))
        assert permutation == Permutation(torch.tensor([0, 2, 1, 3]))
        assert permutation != Permutation(torch.tensor([0, 1, 2, 3]))
        assert permutation != Permutation(torch.tensor([0, 2, 1, 3, 4]))
        assert permutation != "not a Permutation"


class TestPermutationInfo:
    @staticmethod
    def test_eq() -> None:
        lin_a, lin_b = nn.Linear(5, 5), nn.Linear(5, 5)
        module = ModuleBase(lin_a, lin_b)
        permutation_info = PermutationInfo(
            module=module,
            axis=0,
            parameter_a=lin_a.weight,
            parameter_b=lin_b.weight,
        )
        assert permutation_info == PermutationInfo(
            module=module,
            axis=0,
            parameter_a=lin_a.weight,
            parameter_b=lin_b.weight,
        )
        assert permutation_info != PermutationInfo(
            module=module,
            axis=1,
            parameter_a=lin_a.weight,
            parameter_b=lin_b.weight,
        )
        assert permutation_info != PermutationInfo(
            module=module,
            axis=0,
            parameter_a=lin_a.bias,
            parameter_b=lin_b.weight,
        )
        assert permutation_info != PermutationInfo(
            module=module,
            axis=0,
            parameter_a=lin_a.weight,
            parameter_b=lin_b.bias,
        )
        assert permutation_info != "not a PermutationInfo"


def test_base_module() -> None:
    with pytest.raises(TypeError):
        ModuleBase(nn.Linear(5, 5), nn.Conv2d(5, 5, 3))

    with pytest.raises(TypeError):
        ModuleBase("foo", "bar")  # type: ignore[arg-type]

    mb = ModuleBase(nn.Linear(5, 5), nn.Linear(5, 5))
    with pytest.raises(NotImplementedError):
        _ = mb.input_permutation

    with pytest.raises(NotImplementedError):
        mb.input_permutation = Permutation(torch.arange(5))

    with pytest.raises(NotImplementedError):
        _ = mb.output_permutation

    with pytest.raises(NotImplementedError):
        mb.output_permutation = Permutation(torch.arange(5))

    with pytest.raises(NotImplementedError):
        _ = mb.permutation_to_info

    with pytest.raises(NotImplementedError):
        mb.apply_permutations()


class TestDefaultModule:
    @staticmethod
    def test_sanity_checks() -> None:
        with pytest.raises(AttributeError):
            DefaultModule(nn.MultiheadAttention(5, 5), nn.MultiheadAttention(5, 5))

        with pytest.raises(AttributeError):
            lin_a, lin_b = nn.Linear(5, 5), nn.Linear(5, 5)
            del lin_b.weight
            DefaultModule(lin_a, lin_b)

        with pytest.raises(AttributeError):
            lin_a, lin_b = nn.Linear(5, 5), nn.Linear(5, 5)
            del lin_a.bias
            DefaultModule(lin_a, lin_b)

        with pytest.raises(AttributeError):
            lin_a, lin_b = nn.Linear(5, 5), nn.Linear(5, 5)
            del lin_b.bias
            DefaultModule(lin_a, lin_b)

        with pytest.raises(ValueError):
            DefaultModule(nn.Linear(5, 3), nn.Linear(5, 5))

        with pytest.raises(ValueError):
            DefaultModule(nn.Linear(5, 5), nn.Linear(5, 5, bias=False))

        with pytest.raises(ValueError):
            lin_a, lin_b = nn.Linear(5, 5), nn.Linear(5, 5)
            lin_b.bias = nn.Parameter(torch.zeros(3))
            DefaultModule(lin_a, lin_b)

        with pytest.raises(TypeError):
            lin_a, lin_b = nn.Linear(5, 5), nn.Linear(5, 5)
            lin_a.weight = nn.Linear(5, 5)  # type: ignore[assignment]
            DefaultModule(lin_a, lin_b)

        with pytest.raises(TypeError):
            lin_a, lin_b = nn.Linear(5, 5), nn.Linear(5, 5)
            lin_b.weight = nn.Linear(5, 5)  # type: ignore[assignment]
            DefaultModule(lin_a, lin_b)

        with pytest.raises(TypeError):
            lin_a, lin_b = nn.Linear(5, 5), nn.Linear(5, 5)
            lin_a.bias = nn.Linear(5, 5)  # type: ignore[assignment]
            DefaultModule(lin_a, lin_b)

        with pytest.raises(TypeError):
            lin_a, lin_b = nn.Linear(5, 5), nn.Linear(5, 5)
            lin_b.bias = nn.Linear(5, 5)  # type: ignore[assignment]
            DefaultModule(lin_a, lin_b)

    @staticmethod
    def test_io_permutations() -> None:
        mb = initialize_module(nn.Linear(3, 5), nn.Linear(3, 5))
        assert mb.input_permutation == Permutation(torch.arange(3))
        assert mb.output_permutation == Permutation(torch.arange(5))

        mb.input_permutation = Permutation(torch.tensor([2, 0, 1]))
        assert mb.input_permutation == Permutation(torch.tensor([2, 0, 1]))

        mb.output_permutation = Permutation(torch.tensor([4, 2, 3, 1, 0]))
        assert mb.output_permutation == Permutation(torch.tensor([4, 2, 3, 1, 0]))

        # Test 1d weight
        mb = initialize_module(nn.LayerNorm(4), nn.LayerNorm(4))
        assert mb.input_permutation == Permutation(torch.arange(4))
        assert mb.input_permutation is mb.output_permutation

        mb.input_permutation = Permutation(torch.tensor([2, 0, 1, 3]))
        assert mb.input_permutation == Permutation(torch.tensor([2, 0, 1, 3]))

        # Test nd LayerNorm
        mb = initialize_module(nn.LayerNorm([4, 5]), nn.LayerNorm([4, 5]))

        mb.input_permutation = Permutation(torch.tensor([2, 0, 1, 3]))
        assert mb.input_permutation == Permutation(torch.tensor([2, 0, 1, 3]))

        mb.output_permutation = Permutation(torch.tensor([2, 0, 1, 3, 4]))
        assert mb.output_permutation is None

    @staticmethod
    def test_permutation_to_info() -> None:
        mb = initialize_module(nn.Linear(3, 5), nn.Linear(3, 5))
        mb.input_permutation = Permutation(torch.tensor([2, 0, 1]))
        mb.output_permutation = Permutation(torch.tensor([4, 2, 3, 1, 0]))

        info = mb.permutation_to_info
        assert info == [
            (
                mb.output_permutation,
                [
                    PermutationInfo(
                        mb, 0,
                        mb.module_a.weight, mb.module_b.weight  # type: ignore[arg-type]
                    ),
                    PermutationInfo(
                        mb, 0,
                        mb.module_a.bias, mb.module_b.bias  # type: ignore[arg-type]
                    ),
                ]
            ),
            (
                mb.input_permutation,
                [PermutationInfo(
                    mb, 1,
                    mb.module_a.weight, mb.module_b.weight  # type: ignore[arg-type]
                )]
            )
        ]

        mb = initialize_module(nn.Linear(3, 5, bias=False), nn.Linear(3, 5, bias=False))
        mb.input_permutation = Permutation(torch.tensor([2, 0, 1]))
        mb.output_permutation = Permutation(torch.tensor([4, 2, 3, 1, 0]))

        info = mb.permutation_to_info
        assert info == [
            (
                mb.output_permutation,
                [PermutationInfo(
                    mb, 0,
                    mb.module_a.weight, mb.module_b.weight  # type: ignore[arg-type]
                )]
            ),
            (
                mb.input_permutation,
                [PermutationInfo(
                    mb, 1,
                    mb.module_a.weight, mb.module_b.weight  # type: ignore[arg-type]
                )]
            )
        ]

        # What if the input-and output-Permutation is the same?
        mb = initialize_module(nn.Linear(5, 5), nn.Linear(5, 5))
        mb.input_permutation = Permutation(torch.tensor([2, 0, 1, 3, 4]))
        mb.output_permutation = mb.input_permutation

        info = mb.permutation_to_info
        assert info == [
            (
                mb.input_permutation,
                [
                    PermutationInfo(
                        mb, 0,
                        mb.module_a.weight, mb.module_b.weight  # type: ignore[arg-type]
                    ),
                    PermutationInfo(
                        mb, 0,
                        mb.module_a.bias, mb.module_b.bias  # type: ignore[arg-type]
                    ),
                    PermutationInfo(
                        mb, 1,
                        mb.module_a.weight, mb.module_b.weight  # type: ignore[arg-type]
                    ),
                ]
            )
        ]

    @staticmethod
    def test_apply_permutations() -> None:
        lin_a, lin_b = nn.Linear(3, 5), nn.Linear(3, 5)
        lin_b_orig = copy.deepcopy(lin_b)
        mb = initialize_module(lin_a, lin_b)

        mb.input_permutation = Permutation(torch.tensor([2, 0, 1]))
        mb.output_permutation = Permutation(torch.tensor([4, 2, 3, 1, 0]))
        mb.apply_permutations(except_axis=0)

        assert not torch.allclose(lin_b.weight, lin_b_orig.weight)
        assert torch.allclose(
            lin_b.weight[:, torch.argsort(mb.input_permutation.perm_indices)],
            lin_b_orig.weight
        )
        assert torch.allclose(lin_b.bias, lin_b_orig.bias)

        mb.apply_permutations(except_axis=1)
        assert not torch.allclose(lin_b.bias, lin_b_orig.bias)

        # Does except_axis=-1 mean that permutations are applied to all axes?
        mb.input_permutation = Permutation(
            torch.argsort(mb.input_permutation.perm_indices)
        )
        mb.output_permutation = Permutation(
            torch.argsort(mb.output_permutation.perm_indices)
        )

        mb.apply_permutations()
        assert torch.allclose(lin_b.weight, lin_b_orig.weight)
        assert torch.allclose(lin_b.bias, lin_b_orig.bias)

    @staticmethod
    def test_permute_axis() -> None:
        parameter = nn.Parameter(torch.randn(3, 3))
        parameter_orig = copy.deepcopy(parameter)

        permutation = torch.tensor([2, 0, 1])
        DefaultModule.permute_parameter(parameter, 0, permutation)

        assert not torch.allclose(parameter, parameter_orig)
        assert torch.allclose(
            parameter[torch.argsort(permutation)],
            parameter_orig
        )

    @staticmethod
    def test_none_permuation() -> None:
        lin_a, lin_b = nn.Linear(3, 5), nn.Linear(3, 5)
        lin_b_orig = copy.deepcopy(lin_b)
        mb = initialize_module(lin_a, lin_b)
        mb.input_permutation = None
        mb.output_permutation = None

        mb.apply_permutations()
        assert torch.allclose(lin_b.weight, lin_b_orig.weight)

        mb.input_permutation = Permutation(torch.tensor([2, 0, 1]))
        mb.apply_permutations()
        assert torch.allclose(
            lin_b.weight[:, torch.argsort(mb.input_permutation.perm_indices)],
            lin_b_orig.weight
        )

        mb.input_permutation = Permutation(
            torch.argsort(mb.input_permutation.perm_indices)
        )
        mb.output_permutation = Permutation(torch.tensor([4, 2, 3, 1, 0]))
        mb.apply_permutations()

        assert torch.allclose(
            lin_b.weight[torch.argsort(mb.output_permutation.perm_indices)],
            lin_b_orig.weight
        )


class TestMultiheadAttentionModule:
    @staticmethod
    def test_permutation_to_info() -> None:
        embed_dim = 6
        num_heads = 3
        mha = nn.MultiheadAttention(embed_dim, num_heads)

        mb = MultiheadAttentionModule(mha, mha)
        mb.input_permutation = Permutation(torch.tensor([2, 0, 1, 3, 4, 5]))
        mb.output_permutation = Permutation(torch.tensor([4, 2, 3, 1, 0, 5]))

        info = mb.permutation_to_info
        assert info == [
            (
                mb.input_permutation,
                [PermutationInfo(
                    mb, 1,
                    mb.module_a.in_proj_weight,  # type: ignore[arg-type]
                    mb.module_b.in_proj_weight  # type: ignore[arg-type]
                )]
            ),
            (
                mb.output_permutation,
                [
                    PermutationInfo(
                        mb, 0,
                        mb.module_a  # type: ignore[arg-type, union-attr]
                        .out_proj.weight,
                        mb.module_b  # type: ignore[arg-type, union-attr]
                        .out_proj.weight
                    ),
                    PermutationInfo(
                        mb, 0,
                        mb.module_a.out_proj.bias,  # type: ignore[arg-type, union-attr]
                        mb.module_b.out_proj.bias  # type: ignore[arg-type, union-attr]
                    ),
                ]
            ),
        ]

    @staticmethod
    def test_input_permutation() -> None:
        embed_dim = 6
        num_heads = 3
        mha = nn.MultiheadAttention(embed_dim, num_heads, bias=False)
        mha_orig = copy.deepcopy(mha)

        mb = MultiheadAttentionModule(mha, mha)
        perm = Permutation(torch.tensor([2, 0, 1, 3, 4, 5]))
        mb.input_permutation = perm
        mb.output_permutation = perm

        assert mb.input_permutation == perm
        assert mb.output_permutation == perm
        mb.apply_permutations()
        assert model_change_percent(mha_orig, mha) > 0.1

        mha = nn.MultiheadAttention(embed_dim, num_heads, bias=False, kdim=4, vdim=5)
        mha_orig = copy.deepcopy(mha)
        mb = MultiheadAttentionModule(mha, mha)
        perm = Permutation(torch.tensor([2, 0, 1, 3, 4, 5]))
        mb.input_permutation = perm
        mb.output_permutation = None

        assert mb.input_permutation is None
        assert mb.output_permutation is None
        mb.apply_permutations()

        assert model_change_percent(mha_orig, mha) < 1e-6

    @staticmethod
    def test_io() -> None:
        embed_dim = 6
        num_heads = 3

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mha = nn.MultiheadAttention(embed_dim, num_heads)
                self.lin1 = nn.Linear(embed_dim, embed_dim)
                self.lin2 = nn.Linear(embed_dim, embed_dim)
                self.relu = nn.ReLU()

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                input_tensor = self.relu(self.lin1(input_tensor))
                input_tensor = self.relu(
                    self.mha(input_tensor, input_tensor, input_tensor)[0]
                )
                input_tensor = self.relu(self.lin2(input_tensor))
                return input_tensor

        model = Model()
        model_orig = copy.deepcopy(model)

        x = torch.randn(embed_dim, embed_dim)
        y_orig = model(x)

        lin1_mod = initialize_module(model.lin1, model.lin1)
        mha_mod = initialize_module(model.mha, model.mha)
        lin2_mod = initialize_module(model.lin2, model.lin2)

        perm1 = Permutation(torch.tensor([2, 0, 1, 3, 4, 5]))
        perm2 = Permutation(torch.tensor([0, 4, 2, 5, 1, 3]))

        lin1_mod.output_permutation = perm1
        mha_mod.input_permutation = perm1
        mha_mod.output_permutation = perm2
        lin2_mod.input_permutation = perm2

        lin1_mod.apply_permutations()
        mha_mod.apply_permutations()
        lin2_mod.apply_permutations()

        assert model_change_percent(model_orig, model) > 0.1

        y_new = model(x)
        assert torch.allclose(y_orig, y_new)
