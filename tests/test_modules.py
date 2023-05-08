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

    with pytest.raises(NotImplementedError):
        mb.permute_parameter(
            mb.module_a.weight,  # type: ignore[arg-type]
            0,
            torch.randperm(5),
        )


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
        assert mb.output_permutation == Permutation(torch.arange(5))

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
