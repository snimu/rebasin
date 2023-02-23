from __future__ import annotations

import copy

import torch

from rebasin import PermutationCoordinateDescent

from .fixtures.models import MLP


def test__init__() -> None:
    pcd = PermutationCoordinateDescent(
        MLP(features=100, num_layers=5),
        MLP(features=100, num_layers=5),
    )

    for (w1, w2), permutation in zip(pcd.weights, pcd.wperms, strict=True):
        assert torch.all(w1 == w1 @ permutation)
        assert torch.all(w2 == w2 @ permutation)


def test_coordinate_descent() -> None:
    pcd = PermutationCoordinateDescent(
        MLP(features=10, num_layers=5),
        MLP(features=10, num_layers=5),
    )
    pcd.coordinate_descent()


def test_rebasin() -> None:
    pcd = PermutationCoordinateDescent(
        MLP(features=10, num_layers=5),
        MLP(features=10, num_layers=5),
    )
    pcd.coordinate_descent()

    old_weights = []
    for module in pcd.model2.modules():
        if hasattr(module, "weight"):
            old_weights.append(copy.deepcopy(module.weight))

    pcd.rebasin()

    new_weights = []
    for module in pcd.model2.modules():
        if hasattr(module, "weight"):
            new_weights.append(module.weight)

    diff_acc = 0.0
    for ow, nw in zip(old_weights, new_weights, strict=True):
        diff_acc += float((ow - nw).abs().sum())  # type: ignore[operator]

    assert diff_acc > 0.0
