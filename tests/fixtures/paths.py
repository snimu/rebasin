from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rebasin.initialization._permutation import ModuleParameters, Perm


class ModuleGenerator:
    @property
    def linear3_3(self) -> tuple[nn.Linear, ModuleParameters]:
        linear = nn.Linear(3, 3)
        linear_mp = ModuleParameters(
            linear.weight,
            linear.weight,
            "linear1",
            {0: Perm(torch.arange(3)), 1: Perm(torch.arange(3))}
        )
        return linear, linear_mp

    @property
    def linear3_4(self) -> tuple[nn.Linear, ModuleParameters]:
        linear = nn.Linear(3, 4)
        linear_mp = ModuleParameters(
            linear.weight,
            linear.weight,
            "linear1",
            {0: Perm(torch.arange(4)), 1: Perm(torch.arange(3))}
        )
        return linear, linear_mp

    @property
    def linear4_4(self) -> tuple[nn.Linear, ModuleParameters]:
        linear = nn.Linear(4, 4)
        linear_mp = ModuleParameters(
            linear.weight,
            linear.weight,
            "linear2",
            {0: Perm(torch.arange(4)), 1: Perm(torch.arange(4))}
        )
        return linear, linear_mp

    @property
    def linear4_3(self) -> tuple[nn.Linear, ModuleParameters]:
        linear = nn.Linear(4, 3)
        linear_mp = ModuleParameters(
            linear.weight,
            linear.weight,
            "linear3",
            {0: Perm(torch.arange(3)), 1: Perm(torch.arange(4))}
        )
        return linear, linear_mp

    @property
    def layernorm_3(self) -> tuple[nn.LayerNorm, ModuleParameters]:
        layer_norm = nn.LayerNorm(3)
        layer_norm_mp = ModuleParameters(
            layer_norm.weight,
            layer_norm.bias,
            "layer_norm",
            {0: Perm(torch.arange(3))}
        )
        return layer_norm, layer_norm_mp

    @property
    def layernorm_4(self) -> tuple[nn.LayerNorm, ModuleParameters]:
        layer_norm = nn.LayerNorm(4)
        layer_norm_mp = ModuleParameters(
            layer_norm.weight,
            layer_norm.bias,
            "layer_norm",
            {0: Perm(torch.arange(4))}
        )
        return layer_norm, layer_norm_mp
