from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST  # type: ignore[import]
from torchvision.transforms import ToTensor  # type: ignore[import]


def get_mnist(train: bool = True, batch_size: int = 512) -> DataLoader[Any]:
    mnist = MNIST("./data", download=True, train=train, transform=ToTensor())
    mnist_dl = DataLoader(mnist, batch_size=batch_size, shuffle=True)
    return mnist_dl
