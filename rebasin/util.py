from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import nn
from torch.utils.data import DataLoader


def recalculate_batch_norms(
        model: nn.Module,
        dataloader: DataLoader[Any],
        input_indices: Sequence[int],
        *forward_args: Any,
        **forward_kwargs: Any
) -> None:
    """
    Recalculate the BatchNorm statistics of a model.

    Use this after permuting the model, but only if the model contains
    a BatchNorm.

    Args:
        model:
            The model.
        dataloader:
            The DataLoader to use for recalculating the statistics.
            Should ideally be the training DataLoader.
        input_indices:
            Many DataLoaders return several inputs and outputs.
            These can sometimes be used in unexpected ways.
            To make sure that the correct outputs of the dataloader are used
            as inputs to the model's forward pass, you can specify the indices
            at which the inputs are located, in the order that they should be
            passed to the model.
        *forward_args:
            Any additional positional arguments to pass to the model's forward  pass.
        **forward_kwargs:
            Any additional keyword arguments to pass to the model's forward pass.
    """
    # Don't waste compute on models without BatchNorm
    types = [type(m) for m in model.modules()]
    if (
            nn.BatchNorm1d not in types
            and nn.BatchNorm2d not in types
            and nn.BatchNorm3d not in types
    ):
        return

    training = model.training
    model.train()

    for batch in dataloader:
        inputs = [batch[i] for i in input_indices]
        model(*inputs, *forward_args, **forward_kwargs)

    if not training:
        model.eval()
