from __future__ import annotations

import sys

import pytest
import torch

from rebasin import Interpolation, PermutationCoordinateDescent

from .fixtures.dataloaders import get_mnist
from .fixtures.models import MLP
from .fixtures.training import train


@pytest.mark.skipif(
    "--full-suite" not in sys.argv,
    reason="Too compute-intensive to run every time. "
    "Skipping unless '--full-suite'-option is given explicitly",
)
def test_interpolation() -> None:
    mnist_dl = get_mnist(batch_size=1, train=False)
    loss_fn = torch.nn.CrossEntropyLoss()

    model1: torch.nn.Module = MLP(in_features=784, out_features=1, num_layers=5)
    optimizer1 = torch.optim.Adam(params=model1.parameters(), lr=3e-4)
    model2: torch.nn.Module = MLP(in_features=784, out_features=1, num_layers=5)
    optimizer2 = torch.optim.Adam(params=model2.parameters(), lr=1e-3)

    model1 = train(model1, mnist_dl, loss_fn, optimizer1)
    model2 = train(model2, mnist_dl, loss_fn, optimizer2)

    pcd = PermutationCoordinateDescent(model1, model2)
    pcd.coordinate_descent()
    pcd.rebasin()

    interp = Interpolation(
        model1=pcd.model1,
        model2=pcd.model2,
        dataloader=mnist_dl,
        loss_fn=loss_fn,
        verbose=True,
    )
    interp.interpolate(steps=5)
