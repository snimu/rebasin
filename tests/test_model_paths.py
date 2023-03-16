from __future__ import annotations

import torch
from torchvision.models import resnet18  # type: ignore[import]

from rebasin import ModelPaths

from .fixtures.models import MLP


def test_base_path_mlp() -> None:
    model = MLP(5)
    modules = list(model.modules())
    input_data = torch.randn(5)
    paths = ModelPaths(model, input_data)
    assert len(paths.base_paths) == 1
    assert len(paths.base_paths[0]) >= 1

    for node in paths.base_paths[0]:
        module = paths.id_module_map.get(node.compute_unit_id)
        assert hasattr(module, "weight")
        assert module in modules


def test_base_path_resnet18() -> None:
    model = resnet18()
    modules = list(model.modules())
    input_data = torch.randn(1, 3, 224, 224)
    paths = ModelPaths(model, input_data)

    assert len(paths.base_paths) > 1
    for path in paths.base_paths:
        assert len(path) >= 1

        for node in path:
            module = paths.id_module_map.get(node.compute_unit_id)
            assert hasattr(module, "weight")
            assert module in modules


def test_paths_mlp() -> None:
    model = MLP(5)
    input_data = torch.randn(5)
    paths = ModelPaths(model, input_data)
    assert paths.paths == paths.base_paths


def test_paths_resnet18() -> None:
    model = resnet18()
    input_data = torch.randn(1, 3, 224, 224)
    paths = ModelPaths(model, input_data)
    assert len(paths.paths) > len(paths.base_paths)
