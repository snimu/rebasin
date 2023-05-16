from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torchview import ModuleNode, draw_graph

from rebasin.type_definitions import NODE_TYPES


def model_similarity(model_a: nn.Module, model_b: nn.Module) -> float:
    """Calculate the distance between two models."""
    total_dist = 0.0

    for parameter_a, parameter_b in zip(
            model_a.parameters(), model_b.parameters()
    ):
        p_a, p_b = parameter_a.reshape(-1), parameter_b.reshape(-1)
        p_a_ = p_a.to(p_b.device)
        total_dist += float(torch.abs(p_a_ @ p_b))
    return abs(total_dist)


def model_distance(model_a: nn.Module, model_b: nn.Module) -> float:
    dist = 0.0

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
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
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        diff += torch.sum(torch.abs(p1 - p2)).item()
        base += torch.sum(torch.abs(p2)).item()
    return diff / base


def reset_bn_running_stats(model: nn.Module) -> None:
    """Reset the running statistics of all batch norm modules in the model."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d):
            module.reset_running_stats()


def tensor_diff_perc(
        y_orig: torch.Tensor | nn.Parameter, y_new: torch.Tensor | nn.Parameter
) -> float:
    diff = (y_orig - y_new).abs().sum()
    base = y_orig.abs().sum()
    perc = diff / base
    return perc.item()


def modules_and_module_nodes(
        module_a: nn.Module,
        module_b: nn.Module,
        x: torch.Tensor | Sequence[torch.Tensor]
) -> tuple[nn.Module, nn.Module, ModuleNode]:
    nodes: list[NODE_TYPES] = list(
        draw_graph(module_b, input_data=x, depth=1e12).root_container
    )
    while nodes and not isinstance(nodes[0], ModuleNode):
        nodes = list(nodes[0].children)  # type: ignore[arg-type]

    node: ModuleNode = nodes[0]  # type: ignore[assignment]
    assert isinstance(node, ModuleNode)
    return module_a, module_b, node
