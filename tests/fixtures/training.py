from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


def train(
        model: torch.nn.Module,
        dataloader: DataLoader[Any],
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> torch.nn.Module:
    """Most basic of training --- should achieve a bare minimum of performance."""
    for data, target in iter(dataloader):
        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

    return model
