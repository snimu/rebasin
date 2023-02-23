from __future__ import annotations

import sys

import pytest
import torch

from rebasin import Interpolation, PermutationCoordinateDescent

from .fixtures.dataloaders import get_mnist
from .fixtures.models import MLP


@pytest.mark.skipif(
    "--full-suite" not in sys.argv,
    reason="Too compute-intensive to run every time. "
    "Skipping unless '--full-suite'-option is given explicitly",
)
def test_interpolation() -> None:
    mnist_dl = get_mnist(batch_size=1, train=False)

    pcd = PermutationCoordinateDescent(
        MLP(in_features=784, out_features=1, num_layers=5),
        MLP(in_features=784, out_features=1, num_layers=5),
    )
    pcd.coordinate_descent()
    pcd.rebasin()

    interp = Interpolation(
        model1=pcd.model1,
        model2=pcd.model2,
        dataloader=mnist_dl,
        loss_fn=torch.nn.CrossEntropyLoss(),
        verbose=True,
    )
    interp.interpolate(steps=5)
    import warnings
    warnings.warn(f"{interp.losses=}")
