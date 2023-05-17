from __future__ import annotations

import copy
import sys
from typing import Any

import pytest
import torch
from torch import nn

from rebasin.modules import (  # type: ignore[attr-defined]
    Permutation,
    initialize_module,
)
from rebasin.paths import LinearPath, ParallelPaths, PathSequence
from tests.fixtures.paths import PathSource
from tests.fixtures.utils import (
    allclose,
    model_change_percent,
    tensor_diff_perc,
)


class TestLinearPath(PathSource):
    def test_len(self) -> None:
        _, path = self.dense_lin_path()
        assert len(path) == 4

    def test_iter(self) -> None:
        _, path = self.dense_lin_path()
        assert list(path) == list(path.modules)

    def test_getitem(self) -> None:
        _, path = self.dense_lin_path()
        assert path[0] == path.modules[0]
        assert path[1] == path.modules[1]

    def test_bool(self) -> None:
        _, path = self.dense_lin_path()
        assert bool(path) is True
        assert not LinearPath()

    def test_input_permutation(self) -> None:
        _, lin_path = self.dense_lin_path()
        assert lin_path.input_permutation == lin_path[0].input_permutation

        permutation = Permutation(torch.randperm(10))
        lin_path.input_permutation = permutation
        assert lin_path.input_permutation is permutation
        assert lin_path[1].input_permutation is permutation  # It's a 1d-module

        _, conv_path = self.conv_lin_path()
        assert conv_path.input_permutation == conv_path[0].input_permutation

        permutation = Permutation(torch.randperm(3))
        conv_path.input_permutation = permutation
        assert conv_path.input_permutation is permutation
        assert conv_path[1].input_permutation is permutation  # It's a 2d-module
        assert conv_path[1].output_permutation is not permutation

    def test_output_permutation(self) -> None:
        _, lin_path = self.dense_lin_path()
        assert lin_path.output_permutation == lin_path[-1].output_permutation

        permutation = Permutation(torch.randperm(10))
        lin_path.output_permutation = permutation
        assert lin_path.output_permutation is permutation
        assert lin_path[-1].input_permutation is permutation  # It's a 1d-module
        assert lin_path[-2].output_permutation is permutation  # It's 1d and a LayerNorm

        _, conv_path = self.conv_lin_path()
        assert conv_path.output_permutation == conv_path[-1].output_permutation

        permutation = Permutation(torch.randperm(3))
        conv_path.output_permutation = permutation
        assert conv_path.output_permutation is permutation
        assert conv_path[-1].output_permutation is permutation  # It's a LayerNorm
        assert conv_path[-1].input_permutation is permutation
        assert conv_path[-2].output_permutation is permutation

    def test_io_linear(self) -> None:
        lin_model, lin_path = self.dense_lin_path()
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
        conv_model, conv_path = self.conv_lin_path()
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

    def test_enforce_identity(self) -> None:
        model1, path1 = self.dense_lin_path()
        model2, path2 = self.dense_lin_path()
        model3, path3 = self.dense_lin_path()

        model = nn.Sequential(model1, model2, model3)
        model_orig = copy.deepcopy(model)
        x = torch.randn(10, 10)
        y_orig = model(x)

        # Randomize permutations
        for (perm1, _), (perm2, _), (perm3, _) in zip(
                path1.permutation_to_info,
                path2.permutation_to_info,
                path3.permutation_to_info
        ):
            perm1.perm_indices = torch.randperm(len(perm1.perm_indices))
            perm2.perm_indices = torch.randperm(len(perm2.perm_indices))
            perm3.perm_indices = torch.randperm(len(perm3.perm_indices))

        path1.enforce_identity(prev_path=None, next_path=path2)
        path2.enforce_identity(prev_path=path1, next_path=path3)
        path3.enforce_identity(prev_path=path2, next_path=None)

        assert path1.input_permutation is None
        assert isinstance(path1.output_permutation, Permutation)
        assert isinstance(path2.input_permutation, Permutation)
        assert isinstance(path2.output_permutation, Permutation)
        assert isinstance(path3.input_permutation, Permutation)
        assert path3.output_permutation is None

        path1.apply_permutations()
        path2.apply_permutations()
        path3.apply_permutations()
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)

    def test_shapes(self) -> None:
        model, path = self.dense_lin_path()
        assert path.input_permutation_shape == 10
        assert path.output_permutation_shape == 10

        model, path = self.conv_lin_path()
        assert path.input_permutation_shape == \
               model[0].weight.shape[1]  # type: ignore[index]
        assert path.output_permutation_shape == \
               model[-1].weight.shape[1]  # type: ignore[index]

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


class TestParallelPaths(PathSource):
    def test_iter(self) -> None:
        _, path = self.dense_parallel_path_with_empty_path()
        for linpath in path:
            assert isinstance(linpath, LinearPath)

    def test_len(self) -> None:
        _, path = self.dense_parallel_path_with_empty_path()
        assert len(path) == 3

    def test_getitem(self) -> None:
        _, path = self.dense_parallel_path_with_empty_path()
        assert path[0] == path.paths[0]
        assert path[1] == path.paths[1]
        assert path[2] == path.paths[2]

    def test_input_permutation(self) -> None:
        _, path = self.dense_parallel_path_with_empty_path()
        path.input_permutation = Permutation(torch.randperm(10))
        assert isinstance(path.input_permutation, Permutation)
        path.input_permutation = None
        assert path.input_permutation is None

        _, path = self.dense_parallel_path_diff_shapes()
        path.input_permutation = Permutation(torch.randperm(10))

        assert path.input_permutation is None

    def test_output_permutation(self) -> None:
        _, path = self.dense_parallel_path_with_empty_path()
        assert path.output_permutation is None
        path.output_permutation = Permutation(torch.randperm(10))
        assert isinstance(path.output_permutation, Permutation)
        path.output_permutation = None
        assert path.output_permutation is None

        _, path = self.dense_parallel_path_diff_shapes()
        assert path.output_permutation is None
        path.output_permutation = Permutation(torch.randperm(10))
        assert path.output_permutation is None

    def test_enforce_identity_between_paths_with_empty_path(self) -> None:
        model1, path1 = self.dense_lin_path()
        model2, path2 = self.dense_parallel_path_with_empty_path()
        model3, path3 = self.dense_lin_path()

        model = nn.Sequential(model1, model2, model3)
        model1_orig = copy.deepcopy(model1)
        model2_orig = copy.deepcopy(model2)
        model3_orig = copy.deepcopy(model3)
        model_orig = copy.deepcopy(model)
        x = torch.randn(10, 10)
        y_orig = model(x)

        # Randomize permutations
        for (perm1, _), (perm2, _), (perm3, _) in zip(
                path1.permutation_to_info,
                path2.permutation_to_info,
                path3.permutation_to_info
        ):
            perm1.perm_indices = torch.randperm(len(perm1.perm_indices))
            perm2.perm_indices = torch.randperm(len(perm2.perm_indices))
            perm3.perm_indices = torch.randperm(len(perm3.perm_indices))

        path1.enforce_identity(prev_path=None, next_path=path2)
        path2.enforce_identity(prev_path=path1, next_path=path3)
        path3.enforce_identity(prev_path=path2, next_path=None)

        path1.apply_permutations()
        path2.apply_permutations()
        path3.apply_permutations()
        assert model_change_percent(model1, model1_orig) > 0.1
        assert model_change_percent(model2, model2_orig) > 0.1
        assert model_change_percent(model3, model3_orig) > 0.1
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)

    def test_enforce_identity_between_paths_no_empty_path(self) -> None:
        model1, path1 = self.dense_lin_path()
        model2, path2 = self.dense_parallel_path_no_empty_path()
        model3, path3 = self.dense_lin_path()

        model = nn.Sequential(model1, model2, model3)
        model1_orig = copy.deepcopy(model1)
        model2_orig = copy.deepcopy(model2)
        model3_orig = copy.deepcopy(model3)
        model_orig = copy.deepcopy(model)
        x = torch.randn(10, 10)
        y_orig = model(x)

        # Randomize permutations
        for (perm1, _), (perm2, _), (perm3, _) in zip(
                path1.permutation_to_info,
                path2.permutation_to_info,
                path3.permutation_to_info
        ):
            perm1.perm_indices = torch.randperm(len(perm1.perm_indices))
            perm2.perm_indices = torch.randperm(len(perm2.perm_indices))
            perm3.perm_indices = torch.randperm(len(perm3.perm_indices))

        path1.enforce_identity(prev_path=None, next_path=path2)
        path2.enforce_identity(prev_path=path1, next_path=path3)
        path3.enforce_identity(prev_path=path2, next_path=None)

        path1.apply_permutations()
        path2.apply_permutations()
        path3.apply_permutations()

        assert model_change_percent(model1, model1_orig) > 0.1
        assert model_change_percent(model2, model2_orig) > 0.1
        assert model_change_percent(model3, model3_orig) > 0.1
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)

    def test_enforce_identity_prev_path_none_with_empty_path(self) -> None:
        model1, path1 = self.dense_parallel_path_with_empty_path()
        model2, path2 = self.dense_lin_path()

        model = nn.Sequential(model1, model2)
        model1_orig = copy.deepcopy(model1)
        model2_orig = copy.deepcopy(model2)
        model_orig = copy.deepcopy(model)
        x = torch.randn(10, 10)
        y_orig = model(x)

        # Randomize permutations
        for (perm1, _), (perm2, _) in zip(
                path1.permutation_to_info,
                path2.permutation_to_info
        ):
            perm1.perm_indices = torch.randperm(len(perm1.perm_indices))
            perm2.perm_indices = torch.randperm(len(perm2.perm_indices))

        path1.enforce_identity(prev_path=None, next_path=path2)
        path2.enforce_identity(prev_path=path1, next_path=None)

        path1.apply_permutations()
        path2.apply_permutations()

        assert model_change_percent(model1, model1_orig) > 0.1
        assert model_change_percent(model2, model2_orig) > 0.1
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)

    def test_enforce_identity_prev_path_none_no_empty_path(self) -> None:
        # Setup
        model1, path1 = self.dense_parallel_path_no_empty_path()
        model2, path2 = self.dense_lin_path()
        model = nn.Sequential(model1, model2)

        model1_orig = copy.deepcopy(model1)
        model2_orig = copy.deepcopy(model2)
        model_orig = copy.deepcopy(model)

        x = torch.randn(10, 10)
        y_orig = model(x)

        # Randomize permutations
        for perm1, _ in path1.permutation_to_info:
            perm1.perm_indices = torch.randperm(len(perm1.perm_indices))

        for perm2, _ in path2.permutation_to_info:
            perm2.perm_indices = torch.randperm(len(perm2.perm_indices))

        # Permute
        path1.enforce_identity(prev_path=None, next_path=path2)
        path2.enforce_identity(prev_path=path1, next_path=None)

        path1.apply_permutations()
        path2.apply_permutations()

        # Assertions
        assert model_change_percent(model1, model1_orig) > 0.1
        assert model_change_percent(model2, model2_orig) > 0.1
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)

    def test_enforce_identity_next_path_none_with_empty_path(self) -> None:
        model1, path1 = self.dense_lin_path()
        model2, path2 = self.dense_parallel_path_with_empty_path()
        model = nn.Sequential(model1, model2)

        model1_orig = copy.deepcopy(model1)
        model2_orig = copy.deepcopy(model2)
        model_orig = copy.deepcopy(model)

        x = torch.randn(10, 10)
        y_orig = model(x)

        # Randomize permutations
        for perm1, _ in path1.permutation_to_info:
            perm1.perm_indices = torch.randperm(len(perm1.perm_indices))

        for perm2, _ in path2.permutation_to_info:
            perm2.perm_indices = torch.randperm(len(perm2.perm_indices))

        # Permute
        path1.enforce_identity(prev_path=None, next_path=path2)
        path2.enforce_identity(prev_path=path1, next_path=None)

        path1.apply_permutations()
        path2.apply_permutations()

        # Assertions
        assert model_change_percent(model1, model1_orig) > 0.1
        assert model_change_percent(model2, model2_orig) > 0.1
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)

    def test_enforce_identity_next_path_none_no_empty_path(self) -> None:
        model1, path1 = self.dense_lin_path()
        model2, path2 = self.dense_parallel_path_no_empty_path()
        model = nn.Sequential(model1, model2)

        model1_orig = copy.deepcopy(model1)
        model2_orig = copy.deepcopy(model2)
        model_orig = copy.deepcopy(model)

        x = torch.randn(10, 10)
        y_orig = model(x)

        # Randomize permutations
        for perm1, _ in path1.permutation_to_info:
            perm1.perm_indices = torch.randperm(len(perm1.perm_indices))

        for perm2, _ in path2.permutation_to_info:
            perm2.perm_indices = torch.randperm(len(perm2.perm_indices))

        # Permute
        path1.enforce_identity(prev_path=None, next_path=path2)
        path2.enforce_identity(prev_path=path1, next_path=None)

        path1.apply_permutations()
        path2.apply_permutations()

        # Assertions
        assert model_change_percent(model1, model1_orig) > 0.1
        assert model_change_percent(model2, model2_orig) > 0.1
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)

    def test_enforce_identity_no_other_paths_with_empty_path(self) -> None:
        model, path = self.dense_parallel_path_with_empty_path()
        model_orig = copy.deepcopy(model)
        x = torch.randn(10, 10)
        y_orig = model(x)

        for perm, _ in path.permutation_to_info:
            perm.perm_indices = torch.randperm(len(perm.perm_indices))

        path.enforce_identity(prev_path=None, next_path=None)
        path.apply_permutations()
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)

    def test_enforce_identity_no_other_paths_no_empty_path(self) -> None:
        model, path = self.dense_parallel_path_no_empty_path()
        model_orig = copy.deepcopy(model)
        x = torch.randn(10, 10)
        y_orig = model(x)

        for perm, _ in path.permutation_to_info:
            perm.perm_indices = torch.randperm(len(perm.perm_indices))

        path.enforce_identity(prev_path=None, next_path=None)
        assert path.input_permutation is None
        assert path.output_permutation is None

        path.apply_permutations()
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)


class TestModelPath(PathSource):
    def test_lin_par_lin_no_empty_path(self) -> None:
        model1, path1 = self.dense_lin_path()
        model2, path2 = self.dense_parallel_path_no_empty_path()
        model3, path3 = self.dense_lin_path()
        model = nn.Sequential(model1, model2, model3)
        graph = PathSequence(path1, path2, path3)

        model1_orig = copy.deepcopy(model1)
        model2_orig = copy.deepcopy(model2)
        model3_orig = copy.deepcopy(model3)
        model_orig = copy.deepcopy(model)

        x = torch.randn(10, 10)
        y_orig = model(x)

        # Randomize permutations
        for perm, _ in graph.permutation_to_info:
            perm.perm_indices = torch.randperm(len(perm.perm_indices))

        # Permute
        graph.enforce_identity()
        graph.apply_permutations()

        # Assertions
        assert model_change_percent(model1, model1_orig) > 0.1
        assert model_change_percent(model2, model2_orig) > 0.1
        assert model_change_percent(model3, model3_orig) > 0.1
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)

    def test_lin_par_lin_with_empty_path(self) -> None:
        model1, path1 = self.dense_lin_path()
        model2, path2 = self.dense_parallel_path_with_empty_path()
        model3, path3 = self.dense_lin_path()
        model = nn.Sequential(model1, model2, model3)
        graph = PathSequence(path1, path2, path3)

        model1_orig = copy.deepcopy(model1)
        model2_orig = copy.deepcopy(model2)
        model3_orig = copy.deepcopy(model3)
        model_orig = copy.deepcopy(model)

        x = torch.randn(10, 10)
        y_orig = model(x)

        # Randomize permutations
        for perm, _ in graph.permutation_to_info:
            perm.perm_indices = torch.randperm(len(perm.perm_indices))

        # Permute
        graph.enforce_identity()
        graph.apply_permutations()

        # Assertions
        assert model_change_percent(model1, model1_orig) > 0.1
        assert model_change_percent(model2, model2_orig) > 0.1
        assert model_change_percent(model3, model3_orig) > 0.1
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)

    def test_par_par_par_par(self) -> None:
        model1, path1 = self.dense_parallel_path_no_empty_path()
        model2, path2 = self.dense_parallel_path_with_empty_path()
        model3, path3 = self.dense_parallel_path_no_empty_path()
        model4, path4 = self.dense_parallel_path_with_empty_path()
        model = nn.Sequential(model1, model2, model3, model4)
        graph = PathSequence(path1, path2, path3, path4)

        model1_orig = copy.deepcopy(model1)
        model2_orig = copy.deepcopy(model2)
        model3_orig = copy.deepcopy(model3)
        model4_orig = copy.deepcopy(model4)
        model_orig = copy.deepcopy(model)

        x = torch.randn(10, 10)
        y_orig = model(x)

        # Randomize permutations
        for perm, _ in graph.permutation_to_info:
            perm.perm_indices = torch.randperm(len(perm.perm_indices))

        # Permute
        graph.enforce_identity()
        graph.apply_permutations()

        # Assertions
        assert model_change_percent(model1, model1_orig) > 0.1
        assert model_change_percent(model2, model2_orig) > 0.1
        assert model_change_percent(model3, model3_orig) > 0.1
        assert model_change_percent(model4, model4_orig) > 0.1
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)

    def test_par_par_par_par2(self) -> None:
        model1, path1 = self.dense_parallel_path_with_empty_path()
        model2, path2 = self.dense_parallel_path_with_empty_path()
        model3, path3 = self.dense_parallel_path_with_empty_path()
        model4, path4 = self.dense_parallel_path_no_empty_path()
        model = nn.Sequential(model1, model2, model3, model4)
        graph = PathSequence(path1, path2, path3, path4)

        model1_orig = copy.deepcopy(model1)
        model2_orig = copy.deepcopy(model2)
        model3_orig = copy.deepcopy(model3)
        model4_orig = copy.deepcopy(model4)
        model_orig = copy.deepcopy(model)

        x = torch.randn(10, 10)
        y_orig = model(x)

        # Randomize permutations
        for perm, _ in graph.permutation_to_info:
            perm.perm_indices = torch.randperm(len(perm.perm_indices))

        # Permute
        graph.enforce_identity()
        graph.apply_permutations()

        # Assertions
        assert model_change_percent(model1, model1_orig) > 0.1
        assert model_change_percent(model2, model2_orig) > 0.1
        assert model_change_percent(model3, model3_orig) > 0.1
        assert model_change_percent(model4, model4_orig) > 0.1
        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)
        assert allclose(y_orig, y_new)


@pytest.mark.skipif(
    "--full-suite" not in sys.argv,
    reason="Long test. Run with --full-suite to include."
)
class TestLengtheningModel(PathSource):
    iters = 100

    def test_par_no_empty_path(self) -> None:
        constructors = [self.dense_parallel_path_no_empty_path]
        self.print_lengthening_model(constructors, self.iters)

    def test_par_empty_path(self) -> None:
        constructors = [self.dense_parallel_path_with_empty_path]
        self.print_lengthening_model(constructors, self.iters)

    def test_lin__par_no_empty_path(self) -> None:
        constructors = [self.dense_lin_path, self.dense_parallel_path_no_empty_path]
        self.print_lengthening_model(constructors, self.iters)

    def test_lin__par_empty_path(self) -> None:
        constructors = [self.dense_lin_path, self.dense_parallel_path_with_empty_path]
        self.print_lengthening_model(constructors, self.iters)

    def test_par_no_empty_path__lin(self) -> None:
        constructors = [self.dense_parallel_path_no_empty_path, self.dense_lin_path]
        self.print_lengthening_model(constructors, self.iters)

    def test_par_empty_path__lin(self) -> None:
        constructors = [self.dense_parallel_path_with_empty_path, self.dense_lin_path]
        self.print_lengthening_model(constructors, self.iters)

    def test_par_empty_path__par_no_empty_path(self) -> None:
        constructors = [
            self.dense_parallel_path_with_empty_path,
            self.dense_parallel_path_no_empty_path
        ]
        self.print_lengthening_model(constructors, self.iters)

    @staticmethod
    def print_lengthening_model(constructors: list[Any], iters: int) -> None:
        model = nn.Sequential()
        paths: list[ParallelPaths | LinearPath] = []
        x = torch.randn(10, 10)

        diffs: list[float] = []

        for _ in range(iters):
            for constructor in constructors:
                model_n, path_n = constructor()
                model.append(model_n)
                paths.append(path_n)
            graph = PathSequence(*paths)
            model_orig = copy.deepcopy(model)

            y_orig = model(x)

            # Randomize permutations
            for perm, _ in graph.permutation_to_info:
                perm.perm_indices = torch.randperm(len(perm.perm_indices))

            # Permute
            graph.enforce_identity()
            graph.apply_permutations()

            # Assertions
            assert model_change_percent(model, model_orig) > 0.1

            y_new = model(x)
            diff = tensor_diff_perc(y_new, y_orig)
            diffs.append(diff)
            print(len(model), diff)
            assert allclose(y_orig, y_new)
