from __future__ import annotations

import torch

from rebasin import PermutationCoordinateDescent

from .fixtures.models import MLP


def test_permutation_coordinate_descent_mlp() -> None:
    model_a = MLP(5)
    model_b = MLP(5)
    input_data = torch.randn(5)
    pcd = PermutationCoordinateDescent(model_a, model_b, input_data)
    modules_a = [module for module in model_a.modules() if hasattr(module, "weight")]
    modules_b = [module for module in model_b.modules() if hasattr(module, "weight")]

    for module in modules_a:
        assert module in pcd.id_to_module_a.values()

    for module in modules_b:
        assert module in pcd.id_to_module_b.values()

    assert len(pcd.id_to_module_a) == len(pcd.id_to_module_b)
    assert len(pcd.id_to_module_node_a) == len(pcd.id_to_module_node_b)
    assert len(pcd.id_to_permutation_a) == len(pcd.id_to_permutation_b)
    assert len(pcd.num_to_id_a) == len(pcd.num_to_id_b)

    # pcd.id_to_module_a/b can be longer than the other dicts
    #   because it contains all modules, not just the ones with weights.
    assert len(pcd.num_to_id_a) == len(pcd.id_to_module_node_a)
    assert len(pcd.num_to_id_a) == len(pcd.id_to_permutation_a)
