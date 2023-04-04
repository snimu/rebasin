from __future__ import annotations

import torch
from torch import nn


def model_similarity(model_a: nn.Module, model_b: nn.Module) -> float:
    """Calculate the distance between two models."""
    total_dist = 0.0

    for parameter_a, parameter_b in zip(
            model_a.parameters(), model_b.parameters(), strict=True
    ):
        p_a, p_b = parameter_a.reshape(-1), parameter_b.reshape(-1)
        p_a_ = p_a.to(p_b.device)
        total_dist += float(p_a_ @ p_b)
    return abs(total_dist)
