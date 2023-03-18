from __future__ import annotations

import torch
from torchvision.models import resnet18  # type: ignore[import]

from rebasin import PermutationCoordinateDescent

from .fixtures.models import MLP


def test_permutation_coordinate_descent_mlp() -> None:
    do_test(MLP(5), MLP(5), torch.randn(5))


def test_permutation_coordinate_descent_resnet18() -> None:
    do_test(resnet18(), resnet18(), torch.randn(1, 3, 224, 224))


def do_test(
        model_a: torch.nn.Module, model_b: torch.nn.Module, input_data: torch.Tensor
) -> None:
    pcd = PermutationCoordinateDescent(model_a, model_b, input_data)
    modules_a = [module for module in model_a.modules() if hasattr(module, "weight")]
    modules_b = [module for module in model_b.modules() if hasattr(module, "weight")]

    assert len(modules_a) == len(modules_b)
    assert len(modules_a) == len(pcd.num_to_id_a)

    for module in modules_a:
        assert module in pcd.id_to_module_a.values()

    for module in modules_b:
        assert module in pcd.id_to_module_b.values()

    assert len(pcd.id_to_module_a) == len(pcd.id_to_module_b)
    assert len(pcd.id_to_module_node_a) == len(pcd.id_to_module_node_b)
    assert len(pcd.num_to_id_a) == len(pcd.num_to_id_b)

    # pcd.id_to_module_a/b can be longer than the other dicts
    #   because it contains all modules, not just the ones with weights.
    assert len(pcd.num_to_id_a) == len(pcd.id_to_module_node_a)
    assert len(pcd.num_to_id_a) == len(pcd.id_to_permutation)
