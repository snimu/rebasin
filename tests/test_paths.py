"""Test the rebasin.initialization._paths module."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rebasin.initialization._paths import LinearPath, Path, ResidualPath
from rebasin.initialization._permutation import Perm
from tests.fixtures.paths import ModuleGenerator


def allclose(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a, b, atol=1e-7, rtol=1e-4)


class TestLinearPath(ModuleGenerator):
    def test_manual_merging(self) -> None:
        relu = nn.ReLU()
        lin1, lin1_mp = self.linear3_4
        lin2, lin2_mp = self.linear4_4
        lin3, lin3_mp = self.linear4_3

        lin2_mp.input_permutation = lin1_mp.output_permutation
        lin3_mp.input_permutation = lin2_mp.output_permutation

        model = nn.Sequential(lin1, relu, lin2, relu, lin3, relu)
        x = torch.randn(3, 3)
        y_orig = model(x)

        lin1_mp.apply_permutations(except_axis=1)
        lin2_mp.apply_permutations()
        lin3_mp.apply_permutations(except_axis=0)

        y_new = model(x)

        assert allclose(y_orig, y_new)

    def test_simple_path(self) -> None:
        relu = nn.ReLU()
        lin1, lin1_mp = self.linear3_4
        lin2, lin2_mp = self.linear4_4
        lin3, lin3_mp = self.linear4_3

        model = nn.Sequential(lin1, relu, lin2, relu, lin3, relu)

        x = torch.randn(3, 3)
        y_orig = model(x)

        seq = [lin1_mp, lin2_mp, lin3_mp]
        lp = LinearPath(seq)
        path = Path([lp])
        path.merge_permutations()
        path.apply_permutations()

        y_new = model(x)

        assert allclose(y_orig, y_new)

    def test_ln_before_linear(self) -> None:
        for _ in range(10):
            relu = nn.ReLU()
            lin3_4, lin3_4_mp = self.linear3_4
            lin4_4, lin4_4_mp = self.linear4_4
            lin4_3, lin4_3_mp = self.linear4_3
            ln3, ln3_mp = self.layernorm_3

            model = nn.Sequential(
                ln3, lin3_4, relu, lin4_4, relu, lin4_3, relu
            )

            x = torch.randn(3, 3)
            y_orig = model(x)

            seq = [ln3_mp, lin3_4_mp, lin4_4_mp, lin4_3_mp]
            lp = LinearPath(seq)
            path = Path([lp])
            path.merge_permutations()
            path.apply_permutations()

            y_new = model(x)

            assert allclose(y_orig, y_new)

    def test_ln_after_linear(self) -> None:
        for _ in range(10):
            relu = nn.ReLU()
            lin3_4, lin3_4_mp = self.linear3_4
            lin4_4, lin4_4_mp = self.linear4_4
            lin4_3, lin4_3_mp = self.linear4_3
            ln3, ln3_mp = self.layernorm_3

            model = nn.Sequential(
                lin3_4, relu, lin4_4, relu, lin4_3, relu, ln3
            )

            x = torch.randn(3, 3)
            y_orig = model(x)

            seq = [lin3_4_mp, lin4_4_mp, lin4_3_mp, ln3_mp]
            lp = LinearPath(seq)
            path = Path([lp])
            path.merge_permutations()
            path.apply_permutations()

            y_new = model(x)

            assert allclose(y_orig, y_new)

    def test_lin_ln_lin_manual(self) -> None:
        for _ in range(10):
            relu = nn.ReLU()
            lin3_4 = nn.Linear(3, 4, bias=False)
            lin4_3 = nn.Linear(4, 3, bias=False)
            ln4 = nn.LayerNorm(4)

            model = nn.Sequential(lin3_4, relu, ln4, lin4_3, relu)
            x = torch.randn(3)
            y_orig = model(x)

            perm = torch.tensor([0, 2, 1, 3])
            lin3_4.weight.data = lin3_4.weight.data[perm]
            lin4_3.weight.data = lin4_3.weight.data[:, perm]
            ln4.weight.data = ln4.weight.data[perm]

            y_new = model(x)

            assert allclose(y_orig, y_new)

    def test_lin_ln_lin_manual2(self) -> None:
        for _ in range(10):
            relu = nn.ReLU()
            lin3_4, lin3_4_mp = self.linear3_4
            lin4_3, lin4_3_mp = self.linear4_3
            ln4, ln4_mp = self.layernorm_4

            model = nn.Sequential(lin3_4, relu, ln4, lin4_3, relu)
            x = torch.randn(3)
            y_orig = model(x)

            lin3_4_mp.input_permutation = Perm(torch.tensor([0, 1, 2]))
            ln4_mp.input_permutation = lin3_4_mp.output_permutation
            lin4_3_mp.input_permutation = ln4_mp.output_permutation
            lin4_3_mp.output_permutation = Perm(torch.tensor([0, 1, 2]))

            for mod in [lin3_4_mp, ln4_mp, lin4_3_mp]:
                mod.apply_permutations()

            y_new = model(x)

            assert allclose(y_orig, y_new)

    def test_lin_ln_lin(self) -> None:
        for _ in range(10):
            relu = nn.ReLU()
            lin3_4, lin3_4_mp = self.linear3_4
            lin4_3, lin4_3_mp = self.linear4_3
            ln4, ln4_mp = self.layernorm_4

            model = nn.Sequential(
                lin3_4, relu, ln4, lin4_3, relu
            )

            x = torch.randn(3)
            y_orig = model(x)

            seq = [lin3_4_mp, ln4_mp, lin4_3_mp]
            lp = LinearPath(seq)
            path = Path([lp])
            path.merge_permutations()
            path.apply_permutations()

            y_new = model(x)

            assert allclose(y_orig, y_new)

    def test_path_with_layer_norms(self) -> None:
        for _ in range(10):
            relu = nn.ReLU()
            lin3_4, lin3_4_mp = self.linear3_4
            lin4_4, lin4_4_mp = self.linear4_4
            lin4_3, lin4_3_mp = self.linear4_3
            ln4_one, ln4_one_mp = self.layernorm_4
            ln4_two, ln4_two_mp = self.layernorm_4  # Generate a distinct LayerNorm!

            model = nn.Sequential(
                lin3_4, relu, ln4_one, lin4_4, relu, ln4_two, lin4_3, relu
            )

            x = torch.randn(3, 3)
            y_orig = model(x)

            seq = [lin3_4_mp, ln4_one_mp, lin4_4_mp, ln4_two_mp, lin4_3_mp]
            lp = LinearPath(seq)
            path = Path([lp])
            path.merge_permutations()
            path.apply_permutations()

            y_new = model(x)

            assert allclose(y_orig, y_new)


class TestResidualPath(ModuleGenerator):
    def test_simple_path(self) -> None:
        relu = nn.ReLU()
        lin1, lin1_mp = self.linear3_4
        lin2, lin2_mp = self.linear4_4
        lin3, lin3_mp = self.linear4_3

        class ResBlockLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = lin1
                self.lin2 = lin2
                self.lin3 = lin3
                self.relu = relu

            def forward(self, input_tensor: torch.Tensor) -> Any:
                shortcut = input_tensor
                long_cut = self.lin1(input_tensor)
                long_cut = self.relu(long_cut)
                long_cut = self.lin2(long_cut)
                long_cut = self.relu(long_cut)
                long_cut = self.lin3(long_cut)
                long_cut = self.relu(long_cut)
                return shortcut + long_cut

        model = ResBlockLinear()

        x = torch.randn(3, 3)
        y_orig = model(x)

        seq = [lin1_mp, lin2_mp, lin3_mp]
        rp = ResidualPath(long_path=seq, short_path=[])
        path = Path([rp])
        path.merge_permutations()
        path.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)

    def test_path_with_layer_norms(self) -> None:
        relu = nn.ReLU()
        lin3_4, lin3_4_mp = self.linear3_4
        lin4_4, lin4_4_mp = self.linear4_4
        lin4_3, lin4_3_mp = self.linear4_3
        ln4_one, ln4_mp_one = self.layernorm_4
        ln4_two, ln4_mp_two = self.layernorm_4  # Generate a distinct LayerNorm!

        class ResBlockLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.longpath = nn.Sequential(
                    lin3_4, relu, ln4_one, lin4_4, relu, ln4_two, lin4_3, relu
                )

            def forward(self, input_tensor: torch.Tensor) -> Any:
                return input_tensor + self.longpath(input_tensor)

        model = ResBlockLinear()

        x = torch.randn(3, 3)
        y_orig = model(x)

        seq = [lin3_4_mp, ln4_mp_one, lin4_4_mp, ln4_mp_two, lin4_3_mp]
        rp = ResidualPath(long_path=seq, short_path=[])
        path = Path([rp])
        path.merge_permutations()
        path.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)

    def test_path_with_shortcut(self) -> None:
        relu = nn.ReLU()
        lin3_4, lin3_4_mp = self.linear3_4
        lin4_4, lin4_4_mp = self.linear4_4
        lin4_3, lin4_3_mp = self.linear4_3
        ln4_one, ln4_one_mp = self.layernorm_4
        ln4_two, ln4_two_mp = self.layernorm_4  # Generate a distinct LayerNorm!
        shortcut, shortcut_mp = self.linear3_3

        class ResBlockLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.longpath = nn.Sequential(
                    lin3_4, relu, ln4_one, lin4_4, relu, ln4_two, lin4_3, relu
                )

            def forward(self, input_tensor: torch.Tensor) -> Any:
                return shortcut(input_tensor) + self.longpath(input_tensor)

        model = ResBlockLinear()

        x = torch.randn(3, 3)
        y_orig = model(x)

        long_path = [lin3_4_mp, ln4_one_mp, lin4_4_mp, ln4_two_mp, lin4_3_mp]
        rp = ResidualPath(long_path=long_path, short_path=[shortcut_mp])
        path = Path([rp])
        path.merge_permutations()
        path.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)


class TestCombinedPaths(ModuleGenerator):
    @property
    def linear_path_simple(self) -> tuple[nn.Module, LinearPath]:
        relu = nn.ReLU()
        lin1, lin1_mp = self.linear3_4
        lin2, lin2_mp = self.linear4_4
        lin3, lin3_mp = self.linear4_3
        model = nn.Sequential(lin1, relu, lin2, relu, lin3, relu)
        return model, LinearPath([lin1_mp, lin2_mp, lin3_mp])

    @property
    def linear_path_with_norms(self) -> tuple[nn.Module, LinearPath]:
        relu = nn.ReLU()
        lin1, lin1_mp = self.linear3_4
        lin2, lin2_mp = self.linear4_4
        lin3, lin3_mp = self.linear4_3
        ln1, ln1_mp = self.layernorm_4
        ln2, ln2_mp = self.layernorm_4
        model = nn.Sequential(lin1, relu, ln1, lin2, relu, ln2, lin3, relu)
        return model, LinearPath([lin1_mp, ln1_mp, lin2_mp, ln2_mp, lin3_mp])

    @property
    def residual_path_simple(self) -> tuple[nn.Module, ResidualPath]:
        relu = nn.ReLU()
        lin3_4, lin3_4_mp = self.linear3_4
        lin4_4, lin4_4_mp = self.linear4_4
        lin4_3, lin4_3_mp = self.linear4_3
        ln4_one, ln4_mp_one = self.layernorm_4
        ln4_two, ln4_mp_two = self.layernorm_4  # Generate a distinct LayerNorm!

        class ResBlockLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.longpath = nn.Sequential(
                    lin3_4, relu, ln4_one, lin4_4, relu, ln4_two, lin4_3, relu
                )

            def forward(self, input_tensor: torch.Tensor) -> Any:
                return input_tensor + self.longpath(input_tensor)

        seq = [lin3_4_mp, ln4_mp_one, lin4_4_mp, ln4_mp_two, lin4_3_mp]
        rp = ResidualPath(long_path=seq, short_path=[])
        return ResBlockLinear(), rp

    @property
    def residual_path_with_shortcut(self) -> tuple[nn.Module, ResidualPath]:
        relu = nn.ReLU()
        lin3_4, lin3_4_mp = self.linear3_4
        lin4_4, lin4_4_mp = self.linear4_4
        lin4_3, lin4_3_mp = self.linear4_3
        ln4_one, ln4_one_mp = self.layernorm_4
        ln4_two, ln4_two_mp = self.layernorm_4

        shortcut, shortcut_mp = self.linear3_3

        class ResBlockLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.longpath = nn.Sequential(
                    lin3_4, relu, ln4_one, lin4_4, relu, ln4_two, lin4_3, relu
                )

            def forward(self, input_tensor: torch.Tensor) -> Any:
                return shortcut(input_tensor) + self.longpath(input_tensor)

        seq = [lin3_4_mp, ln4_one_mp, lin4_4_mp, ln4_two_mp, lin4_3_mp]
        rp = ResidualPath(long_path=seq, short_path=[shortcut_mp])
        return ResBlockLinear(), rp

    def test_linear(self) -> None:
        model_lin_simple, linear_path_simple = self.linear_path_simple
        model_lin_norm, linear_path_norm = self.linear_path_with_norms

        model = nn.Sequential(model_lin_simple, model_lin_norm)

        x = torch.randn(3, 3)
        y_orig = model(x)

        path = Path([linear_path_simple, linear_path_norm])

        path.merge_permutations()
        path.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)

    def test_residual(self) -> None:
        model_res_simple, residual_path_simple = self.residual_path_simple
        model_res_shortcut, residual_path_shortcut = self.residual_path_with_shortcut

        model = nn.Sequential(model_res_simple, model_res_shortcut)
        x = torch.randn(3, 3)
        y_orig = model(x)

        path = Path([residual_path_simple, residual_path_shortcut])
        path.merge_permutations()
        path.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)

    def test_simple(self) -> None:
        model_lin, path_lin = self.linear_path_simple
        model_res, path_res = self.residual_path_simple

        model = nn.Sequential(model_lin, model_res)
        x = torch.randn(3, 3)
        y_orig = model(x)

        path = Path([path_lin, path_res])
        path.merge_permutations()
        path.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)

    def test_all(self) -> None:
        model_lin_simple, linear_path_simple = self.linear_path_simple
        model_lin_norm, linear_path_norm = self.linear_path_with_norms
        model_res_simple, residual_path_simple = self.residual_path_simple
        model_res_shortcut, residual_path_shortcut = self.residual_path_with_shortcut

        model = nn.Sequential(
            model_lin_simple,
            model_lin_norm,
            model_res_simple,
            model_res_shortcut,
        )
        x = torch.randn(3, 3)
        y_orig = model(x)

        path = Path(
            [
                linear_path_simple,
                linear_path_norm,
                residual_path_simple,
                residual_path_shortcut,
            ]
        )

        path.merge_permutations()
        path.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)
