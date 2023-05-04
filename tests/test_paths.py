"""Test the rebasin.initialization._paths module."""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
from torch import nn

from rebasin._paths import ModelPaths
from rebasin.structs import ModuleParameters
from tests.fixtures.paths import ModuleGenerator
from tests.fixtures.utils import allclose, model_change_percent


class TestPaths(ModuleGenerator):

    def test_linear_path(self) -> None:
        change_percs = torch.empty(10)

        for i in range(10):
            model, seq = self.linear_path()
            model_orig = copy.deepcopy(model)
            x = torch.randn(3, 3)
            y_orig = model(x)

            path = ModelPaths(seq)
            path.apply_permutations()

            # Check that the model has changed
            change_percs[i] = (model_change_percent(model, model_orig))

            # Check that the output nevertheless stays the same
            y_new = model(x)
            assert allclose(y_orig, y_new)

        assert change_percs.mean().item() > 0.1

    def test_dense_residual_path_simple(self) -> None:
        change_percs = torch.empty(10)

        for i in range(10):
            model, paths = self.residual_path_linear_simple()
            model_orig = copy.deepcopy(model)
            x = torch.randn(3, 3)
            y_orig = model(x)

            path = ModelPaths(paths)
            path.apply_permutations()

            # Check that the model has changed
            change_percs[i] = (model_change_percent(model, model_orig))

            # Check that the output nevertheless stays the same
            y_new = model(x)
            assert allclose(y_orig, y_new)

        assert change_percs.mean().item() > 0.1

    def test_dense_residual_path_complex(self) -> None:
        change_percs = torch.empty(10)

        for i in range(10):
            model, paths = self.residual_path_linear_complex()
            model_orig = copy.deepcopy(model)
            x = torch.randn(3, 3)
            y_orig = model(x)

            path = ModelPaths(paths)
            path.apply_permutations()

            # Check that the model has changed
            change_percs[i] = (model_change_percent(model, model_orig))

            # Check that the output nevertheless stays the same
            y_new = model(x)
            assert allclose(y_orig, y_new)

        assert change_percs.mean().item() > 0.1

    def test_combined_dense_paths(self) -> None:
        change_percs = torch.empty(10)

        for i in range(10):
            model_res_one, paths_res_one = self.residual_path_linear_complex()
            model_res_two, paths_res_two = self.residual_path_linear_simple()
            model_lin, paths_lin = self.linear_path()

            model = nn.Sequential(model_res_one, model_lin, model_res_two)
            model_orig = copy.deepcopy(model)
            x = torch.randn(3, 3)
            y_orig = model(x)

            path = ModelPaths(paths_res_one + paths_lin + paths_res_two)
            path.apply_permutations()

            # Check that the model has changed
            change_percs[i] = (model_change_percent(model, model_orig))

            # Check that the output nevertheless stays the same
            y_new = model(x)
            assert allclose(y_orig, y_new)

        assert change_percs.mean().item() > 0.1

    @staticmethod
    def path_test(
            func: Callable[
                [],
                tuple[torch.Tensor, nn.Module, list[list[ModuleParameters]]]
            ]
    ) -> None:
        change_percs = torch.empty(10)

        for i in range(10):
            x, model, paths = func()
            model_orig = copy.deepcopy(model)
            y_orig = model(x)

            path = ModelPaths(paths)
            path.apply_permutations()

            permstr = "\n".join(
                [
                    f"{p.module_type.__name__}:"
                    f"\n\tin: {p.input_permutation.perm_indices}"
                    f"\n\tout: {p.output_permutation.perm_indices}"
                    for p in path.paths[0]
                ]
            )
            print(f"{func.__name__}:\n" + permstr + "\n\n")

            # Check that the model has changed
            change_percs[i] = (model_change_percent(model, model_orig))

            # Check that the output nevertheless stays the same
            y_new = model(x)
            assert allclose(y_orig, y_new)

        assert change_percs.mean().item() > 0.1

    def test_linear_conv_path_simple(self) -> None:
        self.path_test(self.linear_path_conv2d_simple)

    def test_linear_conv_path_medium(self) -> None:
        self.path_test(self.linear_path_conv2d_medium)

    def test_linear_conv_path_complex(self) -> None:
        self.path_test(self.linear_path_conv2d_complex)

    def test_conv_residual_path_simple(self) -> None:
        self.path_test(self.residual_path_conv2d_simple)

    def test_linear_path_with_batchnorm(self) -> None:
        self.path_test(self.linear_path_with_batch_norm)
