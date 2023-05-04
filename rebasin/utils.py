from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rebasin.structs import ModelInfo


def recalculate_batch_norms(
        model: nn.Module,
        dataloader: DataLoader[Any],
        input_indices: int | Sequence[int],
        device: torch.device | str | None,
        verbose: bool,
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
        device:
            The device on which to run the model.
        verbose:
            Whether to print progress.
        *forward_args:
            Any additional positional arguments to pass to the model's forward  pass.
        **forward_kwargs:
            Any additional keyword arguments to pass to the model's forward pass.
    """
    if verbose:
        print("Recalculating BatchNorm statistics...")
    types = [type(m) for m in model.modules()]
    if (
            nn.BatchNorm1d not in types
            and nn.BatchNorm2d not in types
            and nn.BatchNorm3d not in types
    ):
        if verbose:
            print("No BatchNorm layers found in model.")
        return

    training = model.training
    model.train()

    # Reset the running mean and variance
    for module in model.modules():
        if (
                not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                or not hasattr(module, "running_mean")
                or not hasattr(module, "running_var")
                or module.running_mean is None
                or module.running_var is None
        ):
            continue

        model.running_mean = torch.zeros_like(module.running_mean)
        model.running_var = torch.zeros_like(module.running_var)
    # Recalculate the running mean and variance
    for batch in tqdm(dataloader, disable=not verbose):
        if isinstance(batch, Sequence):
            inputs, _ = get_inputs_labels(batch, input_indices, 0, device)
        else:
            inputs = [batch]
        model(*inputs, *forward_args, **forward_kwargs)

    if not training:
        model.eval()


def get_inputs_labels(
        batch: Any,
        input_indices: int | Sequence[int] = 0,
        label_indices: int | Sequence[int] = 1,
        device: torch.device | str | int | None = None
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
        device:
            The device on which to run the model.

    Returns:
        The inputs and labels.
    """
    if isinstance(input_indices, int):
        input_indices = [input_indices]
    if isinstance(label_indices, int):
        label_indices = [label_indices]

    inputs = (
        [batch[i].to(device) for i in input_indices]
        if device is not None
        else [batch[i] for i in input_indices]
    )
    labels = (
        [batch[i].to(device) for i in label_indices]
        if device is not None
        else [batch[i] for i in label_indices]
    )
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


def model_info(model: nn.Module) -> ModelInfo:
    """
    Get information about a model.

    Args:
        model:
            The model.

    Returns:
        The :class:`ModelInfo` object containing the desired information.
    """
    name = model.__class__.__name__
    contains_batch_norm = any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        for m in model.modules()
    )
    num_parameters = sum(p.numel() for p in model.parameters())
    num_weights = sum(1 if "weight" in n else 0 for n, _ in model.named_parameters())
    num_biases = sum(1 if "bias" in n else 0 for n, _ in model.named_parameters())
    num_elements_per_weight = [
        p.numel() for n, p in model.named_parameters() if "weight" in n
    ]
    num_elements_per_bias = [
        p.numel() for n, p in model.named_parameters() if "bias" in n
    ]

    num_permutable_elements_per_weight = []
    for pname, parameter in model.named_parameters():
        if "weight" in pname:
            shape = parameter.shape
            permutable = shape[0] if len(shape) == 1 else shape[0] * shape[1]
            num_permutable_elements_per_weight.append(permutable)

    num_elements_per_weight_mean = torch.mean(
        torch.tensor(num_elements_per_weight, dtype=torch.float)
    ).item()
    num_elements_per_weight_std = torch.std(
        torch.tensor(num_elements_per_weight, dtype=torch.float)
    ).item()
    num_elements_per_bias_mean = torch.mean(
        torch.tensor(num_elements_per_bias, dtype=torch.float)
    ).item()
    num_elements_per_bias_std = torch.std(
        torch.tensor(num_elements_per_bias, dtype=torch.float)
    ).item()

    return ModelInfo(
        name=name,
        contains_batch_norm=contains_batch_norm,
        num_parameters=num_parameters,
        num_weights=num_weights,
        num_biases=num_biases,
        num_elements_per_weight=num_elements_per_weight,
        num_permutable_elements_per_weight=num_permutable_elements_per_weight,
        num_elements_per_bias=num_elements_per_bias,
        num_elements_per_weight_mean=num_elements_per_weight_mean,
        num_elements_per_weight_std=num_elements_per_weight_std,
        num_elements_per_bias_mean=num_elements_per_bias_mean,
        num_elements_per_bias_std=num_elements_per_bias_std,
    )


def pairwise(iterable: Sequence[Any]) -> Iterator[tuple[Any, Any]]:
    """
    Iterate over a sequence pairwise.

    Args:
        iterable:
            The sequence.

    Yields:
        The pairs.
    """
    try:
        return itertools.pairwise(iterable)
    except AttributeError:
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)  # noqa
