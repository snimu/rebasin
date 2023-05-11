from __future__ import annotations

import torch
from torch import nn

from rebasin._initializer import PermutationInitialization
from rebasin.structs import ModuleParameters


def model_similarity(model_a: nn.Module, model_b: nn.Module) -> float:
    """Calculate the distance between two models."""
    total_dist = 0.0

    for parameter_a, parameter_b in zip(  # noqa
            model_a.parameters(), model_b.parameters()
    ):
        p_a, p_b = parameter_a.reshape(-1), parameter_b.reshape(-1)
        p_a_ = p_a.to(p_b.device)
        total_dist += float(torch.abs(p_a_ @ p_b))
    return abs(total_dist)


def model_distance(model_a: nn.Module, model_b: nn.Module) -> float:
    dist = 0.0

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):  # noqa: B905
        dist += (pa - pb).abs().sum().item()

    return dist


def allclose(a: torch.Tensor, b: torch.Tensor) -> bool:
    """A relaxed torch.allclose: torch.allclose(a, b, atol=1e-3, rtol=1e-3).

    Why these tolerances? I don't care much about absolute differences,
    and a relative difference of .1% is acceptable.

    """
    return torch.allclose(a, b, atol=1e-3, rtol=1e-3)


def model_change_percent(model1: nn.Module, model2: nn.Module) -> float:
    diff = 0.0
    base = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):  # noqa: B905
        diff += torch.sum(torch.abs(p1 - p2)).item()
        base += torch.sum(torch.abs(p2)).item()
    return diff / base


def path_analysis(path: list[ModuleParameters]) -> str:
    """Return a string representation of the permutations in a path."""
    pathstr = ""
    for mp in path:
        pathstr += f"{mp.module_type.__name__}"
        pathstr += f"\n\tin: {mp.input_permutation.perm_indices}"
        pathstr += f"\n\tout: {mp.output_permutation.perm_indices}"
    return pathstr


def randomize_permutations(initializer: PermutationInitialization) -> None:
    """Randomize the permutations in :class:`PermutationInitialization`."""
    for permutation, _ in initializer.perm_to_info:
        permutation.perm_indices = torch.randperm(len(permutation.perm_indices))


def reset_bn_running_stats(model: nn.Module) -> None:
    """Reset the running statistics of all batch norm modules in the model."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()


def tensor_diff_perc(
        y_orig: torch.Tensor | nn.Parameter, y_new: torch.Tensor | nn.Parameter
) -> float:
    diff = (y_orig - y_new).abs().sum()
    base = y_orig.abs().sum()
    perc = diff / base
    return perc.item()
