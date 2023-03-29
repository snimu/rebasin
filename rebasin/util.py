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


def get_inputs_labels(
        batch: Any,
        input_indices: int | Sequence[int] | None = None,
        label_indices: int | Sequence[int] | None = None
) -> tuple[list[Any], list[Any]]:
    """
    Get the inputs and outputs from a batch.

    Args:
        batch:
            The batch.
        input_indices:
            Many DataLoaders return several inputs and labels per batch.
            These can sometimes be used in unexpected ways.
            To make sure that the correct outputs of the dataloader are used
            as inputs to the model's forward pass, you can specify the indices
            at which the inputs are located, in the order that they should be
            passed to the model.
        label_indices:
            Like `input_indices`, but for the labels.

    Returns:
        The inputs and labels.
    """
    assert (
        (input_indices is None and label_indices is None)
        or (input_indices is not None and label_indices is not None)
    ), "Either provide both input- and label-indices, or neither"

    if label_indices is None or input_indices is None:
        x, y = batch
        return [x], [y]

    if isinstance(input_indices, int):
        input_indices = [input_indices]
    if isinstance(label_indices, int):
        label_indices = [label_indices]

    inputs = [batch[i] for i in input_indices]
    labels = [batch[i] for i in label_indices]
    return inputs, labels
