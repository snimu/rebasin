"""Test the rebasin.initialization._paths module."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rebasin.initialization._paths import LinearPath, ResidualPath
from tests.fixtures.paths import ModuleGenerator


class TestLinearPath(ModuleGenerator):
    def test_simple_path(self) -> None:
        relu = nn.ReLU()
        lin1, lin1_mp = self.linear3_4
        lin2, lin2_mp = self.linear4_4
        lin3, lin3_mp = self.linear4_3

        model = nn.Sequential(lin1, relu, lin2, relu, lin3, relu)

        x = torch.randn(3, 3)
        y_orig = model(x)

        path = [lin1_mp, lin2_mp, lin3_mp]
        lp = LinearPath(path)
        lp.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)

    def test_path_with_layer_norms(self) -> None:
        relu = nn.ReLU()
        lin3_4, lin3_4_mp = self.linear3_4
        lin4_4, lin4_4_mp = self.linear4_4
        lin4_3, lin4_3_mp = self.linear4_3
        ln4, ln4_mp = self.layernorm_4

        model = nn.Sequential(
            lin3_4, relu, ln4, relu, lin4_4, relu, ln4, relu, lin4_3, relu
        )

        x = torch.randn(3, 3)
        y_orig = model(x)

        path = [lin3_4_mp, ln4_mp, lin4_4_mp, ln4_mp, lin4_3_mp]
        lp = LinearPath(path)
        lp.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)


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

        path = [lin1_mp, lin2_mp, lin3_mp]
        rp = ResidualPath(long_path=path, short_path=[])
        rp.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)

    def test_path_with_layer_norms(self) -> None:
        relu = nn.ReLU()
        lin3_4, lin3_4_mp = self.linear3_4
        lin4_4, lin4_4_mp = self.linear4_4
        lin4_3, lin4_3_mp = self.linear4_3
        ln4, ln4_mp = self.layernorm_4

        class ResBlockLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = lin3_4
                self.ln1 = ln4
                self.lin2 = lin4_4
                self.ln2 = ln4
                self.lin3 = lin4_3
                self.relu = relu

            def forward(self, input_tensor: torch.Tensor) -> Any:
                shortcut = input_tensor
                long_cut = self.lin1(input_tensor)
                long_cut = self.relu(long_cut)
                long_cut = self.ln1(long_cut)
                long_cut = self.lin2(long_cut)
                long_cut = self.relu(long_cut)
                long_cut = self.ln2(long_cut)
                long_cut = self.lin3(long_cut)
                long_cut = self.relu(long_cut)
                return shortcut + long_cut

        model = ResBlockLinear()

        x = torch.randn(3, 3)
        y_orig = model(x)

        path = [lin3_4_mp, ln4_mp, lin4_4_mp, ln4_mp, lin4_3_mp]
        rp = ResidualPath(long_path=path, short_path=[])
        rp.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)

    def test_path_with_shortcut(self) -> None:
        relu = nn.ReLU()
        lin3_4, lin3_4_mp = self.linear3_4
        lin4_4, lin4_4_mp = self.linear4_4
        lin4_3, lin4_3_mp = self.linear4_3
        ln4, ln4_mp = self.layernorm_4
        shortcut, shortcut_mp = self.linear3_3

        class ResBlockLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = lin3_4
                self.ln1 = ln4
                self.lin2 = lin4_4
                self.ln2 = ln4
                self.lin3 = lin4_3
                self.shortcut = shortcut
                self.relu = relu

            def forward(self, input_tensor: torch.Tensor) -> Any:
                short_cut = self.shortcut(input_tensor)
                long_cut = self.lin1(input_tensor)
                long_cut = self.relu(long_cut)
                long_cut = self.ln1(long_cut)
                long_cut = self.lin2(long_cut)
                long_cut = self.relu(long_cut)
                long_cut = self.ln2(long_cut)
                long_cut = self.lin3(long_cut)
                long_cut = self.relu(long_cut)
                return short_cut + long_cut

        model = ResBlockLinear()

        x = torch.randn(3, 3)
        y_orig = model(x)

        path = [lin3_4_mp, ln4_mp, lin4_4_mp, ln4_mp, lin4_3_mp]
        rp = ResidualPath(long_path=path, short_path=[shortcut_mp])
        rp.apply_permutations()

        y_new = model(x)

        assert torch.allclose(y_orig, y_new)
