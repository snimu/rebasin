# type: ignore
# I do type-checks in code, but mypy doesn't understand that.
# This leads to me getting an error from mypy at almost every line,
# which is annoying.
# Instead of reformatting my file such that mypy doesn't complain,
# I just ignore all the errors and make sure to test thoroughly.

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from rebasin.modules import (
    DefaultModule,
    InputPermIsOutputPermMultiDimModule,
    ModuleBase,
    MultiheadAttentionModule,
    OneDimModule,
    Permutation,
    PermutationInfo,
    initialize_module,
)
from tests.fixtures.utils import (
    allclose,
    model_change_percent,
    modules_and_module_nodes,
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
        lin_a, lin_b, node = modules_and_module_nodes(
            nn.Linear(5, 5), nn.Linear(5, 5), torch.randn(5)
        )
        module = ModuleBase(lin_a, lin_b, node)
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
        ModuleBase(
            *modules_and_module_nodes(
                nn.Linear(5, 5), nn.Conv2d(5, 5, 3), torch.randn(1, 5, 15, 15)
            )
        )

    with pytest.raises(TypeError):
        _, _, node = modules_and_module_nodes(
            nn.Linear(5, 5), nn.Linear(5, 5), torch.randn(5)
        )
        ModuleBase("foo", "bar", node)

    mb = ModuleBase(
        *modules_and_module_nodes(
            nn.Conv2d(5, 5, 3), nn.Conv2d(5, 5, 3), torch.randn(1, 5, 15, 15)
        )
    )
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
        _ = mb.input_permutation_shape

    with pytest.raises(NotImplementedError):
        _ = mb.output_permutation_shape


class TestDefaultModule:
    @staticmethod
    def test_sanity_checks() -> None:
        with pytest.raises(AttributeError):
            m1, m2, node = modules_and_module_nodes(
                nn.MultiheadAttention(5, 5),
                nn.MultiheadAttention(5, 5),
                (torch.randn(5, 5), torch.randn(5, 5), torch.randn(5, 5))
            )
            DefaultModule(m1, m2, node)

        with pytest.raises(ValueError):
            lin_a, lin_b, node = modules_and_module_nodes(
                nn.Linear(5, 3), nn.Linear(5, 5), torch.randn(5)
            )
            DefaultModule(lin_a, lin_b, node)

        with pytest.raises(ValueError):
            lin_a, lin_b, node = modules_and_module_nodes(
                nn.Linear(5, 5), nn.Linear(5, 5, bias=False), torch.randn(5)
            )
            DefaultModule(lin_a, lin_b, node)

        with pytest.raises(ValueError):
            lin_a, lin_b, node = modules_and_module_nodes(
                nn.Linear(5, 5), nn.Linear(5, 5), torch.randn(5)
            )
            lin_b.bias = nn.Parameter(torch.zeros(3))
            DefaultModule(lin_a, lin_b, node)

        with pytest.raises(TypeError):
            lin_a, lin_b, node = modules_and_module_nodes(
                nn.Linear(5, 5), nn.Linear(5, 5), torch.randn(5)
            )
            lin_a.weight = nn.Linear(5, 5)
            DefaultModule(lin_a, lin_b, node)

        with pytest.raises(TypeError):
            lin_a, lin_b, node = modules_and_module_nodes(
                nn.Linear(5, 5), nn.Linear(5, 5), torch.randn(5)
            )
            lin_b.weight = nn.Linear(5, 5)
            DefaultModule(lin_a, lin_b, node)

        with pytest.raises(TypeError):
            lin_a, lin_b, node = modules_and_module_nodes(
                nn.Linear(5, 5), nn.Linear(5, 5), torch.randn(5)
            )
            lin_a.bias = nn.Linear(5, 5)
            DefaultModule(lin_a, lin_b, node)

        with pytest.raises(TypeError):
            lin_a, lin_b, node = modules_and_module_nodes(
                nn.Linear(5, 5), nn.Linear(5, 5), torch.randn(5)
            )
            lin_b.bias = nn.Linear(5, 5)
            DefaultModule(lin_a, lin_b, node)

    @staticmethod
    def test_io_permutations() -> None:
        mb = initialize_module(
            *modules_and_module_nodes(nn.Linear(3, 5), nn.Linear(3, 5), torch.randn(3))
        )
        assert mb.input_permutation == Permutation(torch.arange(3))
        assert mb.output_permutation == Permutation(torch.arange(5))

        mb.input_permutation = Permutation(torch.tensor([2, 0, 1]))
        assert mb.input_permutation == Permutation(torch.tensor([2, 0, 1]))

        mb.output_permutation = Permutation(torch.tensor([4, 2, 3, 1, 0]))
        assert mb.output_permutation == Permutation(torch.tensor([4, 2, 3, 1, 0]))

    @staticmethod
    def test_permutation_to_info() -> None:
        mb = initialize_module(
            *modules_and_module_nodes(nn.Linear(3, 5), nn.Linear(3, 5), torch.randn(3))
        )
        mb.input_permutation = Permutation(torch.tensor([2, 0, 1]))
        mb.output_permutation = Permutation(torch.tensor([4, 2, 3, 1, 0]))

        info = mb.permutation_to_info
        assert info == [
            (
                mb.output_permutation,
                [
                    PermutationInfo(mb, 0, mb.module_a.weight, mb.module_b.weight),
                    PermutationInfo(mb, 0, mb.module_a.bias, mb.module_b.bias),
                ]
            ),
            (
                mb.input_permutation,
                [PermutationInfo(mb, 1, mb.module_a.weight, mb.module_b.weight)]
            )
        ]

        mb = initialize_module(
            *modules_and_module_nodes(
                nn.Linear(3, 5, bias=False), nn.Linear(3, 5, bias=False), torch.randn(3)
            )
        )
        mb.input_permutation = Permutation(torch.tensor([2, 0, 1]))
        mb.output_permutation = Permutation(torch.tensor([4, 2, 3, 1, 0]))

        info = mb.permutation_to_info
        assert info == [
            (
                mb.output_permutation,
                [PermutationInfo(mb, 0, mb.module_a.weight, mb.module_b.weight)]
            ),
            (
                mb.input_permutation,
                [PermutationInfo(mb, 1, mb.module_a.weight, mb.module_b.weight)]
            )
        ]

        # What if the input-and output-Permutation is the same?

        mb = initialize_module(
            *modules_and_module_nodes(nn.Linear(5, 5), nn.Linear(5, 5), torch.randn(5))
        )
        mb.input_permutation = Permutation(torch.tensor([2, 0, 1, 3, 4]))
        mb.output_permutation = mb.input_permutation

        info = mb.permutation_to_info
        assert info == [
            (
                mb.input_permutation,
                [
                    PermutationInfo(mb, 0, mb.module_a.weight, mb.module_b.weight),
                    PermutationInfo(mb, 0, mb.module_a.bias, mb.module_b.bias),
                    PermutationInfo(mb, 1, mb.module_a.weight, mb.module_b.weight),
                ]
            )
        ]

    @staticmethod
    def test_apply_permutations() -> None:
        lin_a, lin_b, node = modules_and_module_nodes(
            nn.Linear(3, 5), nn.Linear(3, 5), torch.randn(3)
        )
        lin_b_orig = copy.deepcopy(lin_b)
        mb = initialize_module(lin_a, lin_b, node)

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
        lin_a, lin_b, node = modules_and_module_nodes(
            nn.Linear(3, 5), nn.Linear(3, 5), torch.randn(3)
        )
        lin_b_orig = copy.deepcopy(lin_b)
        mb = initialize_module(lin_a, lin_b, node)
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

    @staticmethod
    def test_shapes() -> None:
        lin_a, lin_b, node = modules_and_module_nodes(
            nn.Linear(3, 5), nn.Linear(3, 5), torch.randn(3)
        )
        mb = initialize_module(lin_a, lin_b, node)
        assert mb.input_permutation_shape == lin_a.weight.shape[1]
        assert mb.output_permutation_shape == lin_a.weight.shape[0]


class TestOneDimModule:
    @staticmethod
    def test_io_permutations() -> None:
        mb = initialize_module(
            *modules_and_module_nodes(
                nn.BatchNorm2d(3), nn.BatchNorm2d(3), torch.randn(1, 3, 12, 12)
            )
        )
        assert isinstance(mb, OneDimModule)
        assert mb.input_permutation == Permutation(torch.arange(3))
        assert mb.input_permutation is mb.output_permutation

        mb.input_permutation = Permutation(torch.tensor([2, 0, 1]))
        assert mb.input_permutation == Permutation(torch.tensor([2, 0, 1]))

    @staticmethod
    def test_apply_permutations() -> None:
        bn_a, bn_b, node = modules_and_module_nodes(
            nn.BatchNorm2d(3), nn.BatchNorm2d(3), torch.randn(1, 3, 12, 12)
        )
        bn_b.weight.data = torch.randn_like(bn_b.weight.data)
        bn_b.bias.data = torch.randn_like(bn_b.bias.data)
        bn_b_orig = copy.deepcopy(bn_b)

        mb = initialize_module(bn_a, bn_b, node)

        mb.input_permutation = Permutation(torch.tensor([2, 0, 1]))
        assert mb.output_permutation is mb.input_permutation
        mb.apply_permutations()

        assert not torch.allclose(bn_b.weight, bn_b_orig.weight)
        assert not torch.allclose(bn_b.bias, bn_b_orig.bias)

    @staticmethod
    def test_none_permutation() -> None:
        mb = initialize_module(
            *modules_and_module_nodes(
                nn.BatchNorm2d(3), nn.BatchNorm2d(3), torch.randn(1, 3, 12, 12)
            )
        )
        assert isinstance(mb, OneDimModule)
        assert mb.input_permutation == Permutation(torch.arange(3))
        assert mb.input_permutation is mb.output_permutation

        mb.input_permutation = None
        assert mb.input_permutation is None
        assert mb.output_permutation is None

    @staticmethod
    def test_shapes() -> None:
        bn_a, bn_b, node = modules_and_module_nodes(
            nn.BatchNorm2d(3), nn.BatchNorm2d(3), torch.randn(1, 3, 12, 12)
        )
        mb = initialize_module(bn_a, bn_b, node)
        assert mb.input_permutation_shape == bn_a.weight.shape[0]
        assert mb.output_permutation_shape == bn_a.weight.shape[0]


class TestInputPermIsOutputPermMultiDimModule:
    @staticmethod
    def test_permutation_to_info() -> None:
        mb = initialize_module(
            *modules_and_module_nodes(
                nn.LayerNorm([4, 5]), nn.LayerNorm([4, 5]), torch.randn(4, 5)
            )
                     )
        assert isinstance(mb, InputPermIsOutputPermMultiDimModule)

        mb.input_permutation = Permutation(torch.tensor([2, 0, 1, 3]))
        assert mb.input_permutation == Permutation(torch.tensor([2, 0, 1, 3]))

        mb.output_permutation = Permutation(torch.tensor([2, 0, 1, 3, 4]))
        assert mb.output_permutation == Permutation(torch.tensor([2, 0, 1, 3, 4]))

    @staticmethod
    def test_apply_permutations() -> None:
        ln_a, ln_b, node = modules_and_module_nodes(
            nn.LayerNorm([4, 5]), nn.LayerNorm([4, 5]), torch.randn(4, 5)
        )
        ln_b.weight.data = torch.randn_like(ln_b.weight.data)
        ln_b.bias.data = torch.randn_like(ln_b.bias.data)
        ln_b_orig = copy.deepcopy(ln_b)
        mb = initialize_module(ln_a, ln_b, node)

        mb.input_permutation = Permutation(torch.tensor([2, 0, 4, 1, 3]))
        assert mb.output_permutation is mb.input_permutation
        mb.apply_permutations()

        assert not torch.allclose(ln_b.weight, ln_b_orig.weight)
        assert not torch.allclose(ln_b.bias, ln_b_orig.bias)

    @staticmethod
    def test_none_permutation() -> None:
        mb = initialize_module(
            *modules_and_module_nodes(
                nn.LayerNorm([4, 5]), nn.LayerNorm([4, 5]), torch.randn(4, 5)
            )
        )
        assert isinstance(mb, InputPermIsOutputPermMultiDimModule)

        mb.input_permutation = None
        assert mb.input_permutation is None
        assert mb.output_permutation is None

    @staticmethod
    def test_shapes() -> None:
        ln_a, ln_b, node = modules_and_module_nodes(
            nn.LayerNorm([4, 5]), nn.LayerNorm([4, 5]), torch.randn(4, 5)
        )
        mb = initialize_module(ln_a, ln_b, node)
        assert mb.input_permutation_shape == ln_a.weight.shape[1]
        assert mb.output_permutation_shape == ln_a.weight.shape[1]

        ln_a, ln_b, node = modules_and_module_nodes(
            nn.LayerNorm(3), nn.LayerNorm(3), torch.randn(4, 3)
        )
        mb = initialize_module(ln_a, ln_b, node)
        assert mb.input_permutation_shape == ln_a.weight.shape[0]
        assert mb.output_permutation_shape == ln_a.weight.shape[0]

    @staticmethod
    def test_embedding() -> None:
        emb_a, emb_b, node = modules_and_module_nodes(
            nn.Embedding(10, 5), nn.Embedding(10, 5), torch.randint(0, 10, (5,))
        )
        emb_mod = initialize_module(emb_a, emb_b, node)

        lin_a, lin_b, node = modules_and_module_nodes(
            nn.Linear(5, 5), nn.Linear(5, 5), torch.randn(4, 5)
        )
        lin_mod = initialize_module(lin_a, lin_b, node)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = lin_b
                self.emb = emb_b
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.relu(self.lin(self.emb(x)))

        model = Model()
        model_orig = copy.deepcopy(model)
        x = torch.randint(0, 10, (5,))
        y_orig = model(x)
        perm = Permutation(torch.tensor([2, 0, 1, 3, 4]))
        emb_mod.output_permutation = perm
        emb_mod.apply_permutations()
        lin_mod.input_permutation = perm
        lin_mod.output_permutation = None
        lin_mod.apply_permutations()

        assert model_change_percent(model, model_orig) > 0.1
        y_new = model(x)
        assert allclose(y_new, y_orig)


class TestMultiheadAttentionModule:
    @staticmethod
    def test_permutation_to_info() -> None:
        embed_dim = 6
        num_heads = 3
        x = torch.randn(6, 6, 6)
        mha, mhb, node = modules_and_module_nodes(
            nn.MultiheadAttention(embed_dim, num_heads),
            nn.MultiheadAttention(embed_dim, num_heads),
            (x, x, x),
        )

        mb = MultiheadAttentionModule(mha, mhb, node)
        mb.input_permutation = Permutation(torch.tensor([2, 0, 1, 3, 4, 5]))
        mb.output_permutation = Permutation(torch.tensor([4, 2, 3, 1, 0, 5]))

        info = mb.permutation_to_info
        assert info == [
            (
                mb.input_permutation,
                [PermutationInfo(
                    mb, 1, mb.module_a.in_proj_weight, mb.module_b.in_proj_weight
                )]
            ),
            (
                mb.output_permutation,
                [
                    PermutationInfo(
                        mb, 0,
                        mb.module_a.out_proj.weight,
                        mb.module_b.out_proj.weight
                    ),
                    PermutationInfo(
                        mb, 0,
                        mb.module_a.out_proj.bias,
                        mb.module_b.out_proj.bias
                    ),
                ]
            ),
        ]

    @staticmethod
    def test_input_permutation() -> None:
        embed_dim = 6
        num_heads = 3
        x = torch.randn(6, 6)
        mha, mhb, node = modules_and_module_nodes(
            nn.MultiheadAttention(embed_dim, num_heads, bias=False),
            nn.MultiheadAttention(embed_dim, num_heads, bias=False),
            (x, x, x),
        )
        mhb_orig = copy.deepcopy(mhb)

        mb = MultiheadAttentionModule(mha, mhb, node)
        perm = Permutation(torch.tensor([2, 0, 1, 3, 4, 5]))
        mb.input_permutation = perm
        mb.output_permutation = perm

        assert mb.input_permutation == perm
        assert mb.output_permutation == perm
        mb.apply_permutations()
        assert model_change_percent(mhb_orig, mhb) > 0.1

        mha, mhb, node = modules_and_module_nodes(
            nn.MultiheadAttention(embed_dim, num_heads, bias=False, kdim=4, vdim=5),
            nn.MultiheadAttention(embed_dim, num_heads, bias=False, kdim=4, vdim=5),
            (x, torch.randn(6, 4), torch.randn(6, 5)),
        )
        mhb_orig = copy.deepcopy(mhb)
        mb = MultiheadAttentionModule(mha, mhb, node)
        perm = Permutation(torch.tensor([2, 0, 1, 3, 4, 5]))
        mb.input_permutation = perm
        mb.output_permutation = None

        assert mb.input_permutation is None
        assert mb.output_permutation is None
        mb.apply_permutations()

        assert model_change_percent(mhb_orig, mhb) < 1e-6

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

        lin1_mod = initialize_module(
            *modules_and_module_nodes(model.lin1, model.lin1, x)
        )
        mha_mod = initialize_module(
            *modules_and_module_nodes(
                model.mha, model.mha, (model.lin1(x), model.lin1(x), model.lin1(x))
            )
        )
        lin2_mod = initialize_module(
            *modules_and_module_nodes(model.lin2, model.lin2, x)
        )

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

    @staticmethod
    def test_shapes() -> None:
        embed_dim = 6
        num_heads = 3
        x = torch.randn(6, 6)
        mha, mhb, node = modules_and_module_nodes(
            nn.MultiheadAttention(embed_dim, num_heads),
            nn.MultiheadAttention(embed_dim, num_heads),
            (x, x, x),
        )
        mb = initialize_module(mha, mhb, node)
        assert isinstance(mhb.in_proj_weight, nn.Parameter)
        assert mb.input_permutation_shape == embed_dim
        assert mb.output_permutation_shape == embed_dim

        mha, mhb, node = modules_and_module_nodes(
            nn.MultiheadAttention(embed_dim, num_heads, kdim=4, vdim=5),
            nn.MultiheadAttention(embed_dim, num_heads, kdim=4, vdim=5),
            (x, torch.randn(6, 4), torch.randn(6, 5)),
        )
        mb = initialize_module(mha, mhb, node)
        assert mhb.in_proj_weight is None
        assert mb.input_permutation_shape == 0
        assert mb.output_permutation_shape == embed_dim
