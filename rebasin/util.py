from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from torch import nn
from torch.utils.data import DataLoader


def recalculate_batch_norms(
        model: nn.Module,
        dataloader: DataLoader[Any],
        input_indices: int | Sequence[int],
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
            Many DataLoaders return several inputs and labels.
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
        if isinstance(batch, Sequence):
            inputs, _ = get_inputs_labels(batch, input_indices, 0)
        else:
            inputs = [batch]
        model(*inputs, *forward_args, **forward_kwargs)

    if not training:
        model.eval()


def get_inputs_labels(
        batch: Any,
        input_indices: int | Sequence[int] = 0,
        label_indices: int | Sequence[int] = 1
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
    if isinstance(input_indices, int):
        input_indices = [input_indices]
    if isinstance(label_indices, int):
        label_indices = [label_indices]

    inputs = [batch[i] for i in input_indices]
    labels = [batch[i] for i in label_indices]
    return inputs, labels


def contains_parameter(
        parameters: Sequence[nn.Parameter] | Iterator[nn.Parameter],
        parameter: nn.Parameter
) -> bool:
    """
    Check if a sequence of parameters contains a parameter.

    This cannot be done via the normal `in` operator, because
    `nn.Parameter`'s `__eq__` does not work for parameters of different shapes.

    Args:
        parameters:
            The sequence of parameters.
        parameter:
            The parameter.

    Returns:
        Whether the sequence contains the parameter.
    """
    return any(param is parameter for param in parameters)
