from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rebasin.modules import initialize_module  # type: ignore[attr-defined]
from rebasin.paths import LinearPath, ParallelPaths, PathSequence
from tests.fixtures.utils import modules_and_module_nodes


class PathSource:
    @staticmethod
    def dense_lin_path() -> tuple[nn.Module, LinearPath]:
        ln1 = nn.LayerNorm(10)
        lin1 = nn.Linear(10, 10)
        lin2 = nn.Linear(10, 10)
        ln2 = nn.LayerNorm(10)

        model = nn.Sequential(ln1, lin1, nn.ReLU(), lin2, nn.ReLU(), ln2)

        x = torch.randn(10)
        mod0 = initialize_module(*modules_and_module_nodes(ln1, ln1, x))
        mod1 = initialize_module(*modules_and_module_nodes(lin1, lin1, x))
        mod2 = initialize_module(*modules_and_module_nodes(lin2, lin2, x))
        mod3 = initialize_module(*modules_and_module_nodes(ln2, ln2, x))

        path = LinearPath(mod0, mod1, mod2, mod3)
        return model, path

    @staticmethod
    def conv_lin_path() -> tuple[nn.Module, LinearPath]:
        ln1 = nn.LayerNorm([3, 10, 10])
        conv1 = nn.Conv2d(3, 3, (3, 3))
        conv2 = nn.Conv2d(3, 3, (3, 3))
        ln2 = nn.LayerNorm([3, 6, 6])

        model = nn.Sequential(ln1, conv1, nn.ReLU(), conv2, nn.ReLU(), ln2)

        x = torch.randn(1, 3, 10, 10)
        mod0 = initialize_module(*modules_and_module_nodes(ln1, ln1, x))
        mod1 = initialize_module(*modules_and_module_nodes(conv1, conv1, ln1(x)))
        mod2 = initialize_module(*modules_and_module_nodes(conv2, conv2, conv1(ln1(x))))
        mod3 = initialize_module(
            *modules_and_module_nodes(ln2, ln2, conv2(conv1(ln1(x))))
        )

        path = LinearPath(mod0, mod1, mod2, mod3)
        return model, path

    @classmethod
    def dense_parallel_path_with_empty_path(cls) -> tuple[nn.Module, ParallelPaths]:
        mod1, path1 = cls.dense_lin_path()
        mod2, path2 = cls.dense_lin_path()
        mod3, path3 = nn.Sequential(), LinearPath()

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod1 = mod1
                self.mod2 = mod2
                self.mod3 = mod3

            def forward(self, x: torch.Tensor) -> Any:
                return self.mod1(x) + self.mod2(x) + self.mod3(x)

        path = ParallelPaths(path1, path2, path3)
        return Model(), path

    @classmethod
    def dense_parallel_path_no_empty_path(cls) -> tuple[nn.Module, ParallelPaths]:
        mod1, path1 = cls.dense_lin_path()
        mod2, path2 = cls.dense_lin_path()

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod1 = mod1
                self.mod2 = mod2

            def forward(self, x: torch.Tensor) -> Any:
                return self.mod1(x) + self.mod2(x)

        path = ParallelPaths(path1, path2)
        return Model(), path

    @classmethod
    def dense_parallel_path_diff_shapes(cls) -> tuple[nn.Module, ParallelPaths]:
        mod1, path1 = cls.dense_lin_path()
        mod2 = nn.Linear(2, 2)
        path2 = LinearPath(
            initialize_module(*modules_and_module_nodes(mod2, mod2, torch.rand(2)))
        )

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod1 = mod1
                self.mod2 = mod2

            def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Any:
                return self.mod1(x1), self.mod2(x2)

        path = ParallelPaths(path1, path2)
        return Model(), path
