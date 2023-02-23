from __future__ import annotations

import torch

from rebasin import PermutationCoordinateDescent

from .fixtures.models import MLP


def test__init__() -> None:
    pcd = PermutationCoordinateDescent(
        MLP(features=10, num_layers=5),
        MLP(features=10, num_layers=5),
    )

    for (w1, w2), permutation in zip(pcd.weights, pcd.wperms, strict=True):
        assert torch.all(w1 == w1 @ permutation)
        assert torch.all(w2 == w2 @ permutation)
