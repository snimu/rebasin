from __future__ import annotations

import torch
from torch.utils.data import Dataset


class RandomDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, shape: torch.Size | tuple[int, ...], length: int) -> None:
        self.len = length
        self.x = [torch.randn(shape) for _ in range(length)]
        self.y = [torch.randn(1) for _ in range(length)]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return self.len
