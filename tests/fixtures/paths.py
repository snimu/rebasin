from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rebasin.initialization._permutation import ModuleParameters, Perm


class ModuleGenerator:
    relu = nn.ReLU()

    @staticmethod
    def linear(
            in_channels: int, out_channels: int, bias: bool = False
    ) -> tuple[nn.Linear, ModuleParameters]:
        linear = nn.Linear(in_channels, out_channels, bias=bias)
        bias_ = linear.bias if bias else None
        linear_mp = ModuleParameters(
            weight_a=linear.weight,
            weight_b=linear.weight,
            name="linear",
            axis_to_permutation={
                0: Perm(torch.randperm(out_channels)),
                1: Perm(torch.randperm(in_channels))
            },
            bias_a=bias_,
            bias_b=bias_,
            module_type=nn.Linear,
        )
        return linear, linear_mp

    @staticmethod
    def layernorm(
            shape: int | list[int] | torch.Size
    ) -> tuple[nn.LayerNorm, ModuleParameters]:
        layer_norm = nn.LayerNorm(shape)
        layer_norm.weight.data *= torch.randn_like(layer_norm.weight.data)

        if isinstance(shape, int):
            axis_to_perm = {0: Perm(torch.randperm(shape))}
        else:
            assert len(shape) > 1, "Shape must be an int or a tuple of length > 1"
            axis_to_perm = {
                0: Perm(torch.randperm(shape[0])),
                1: Perm(torch.randperm(shape[1]))
            }

        layer_norm_mp = ModuleParameters(
            weight_a=layer_norm.weight,
            weight_b=layer_norm.weight,
            name="layer_norm",
            axis_to_permutation=axis_to_perm,
            module_type=nn.LayerNorm,
        )
        return layer_norm, layer_norm_mp

    @staticmethod
    def batchnorm2d(num_features: int) -> tuple[nn.BatchNorm2d, ModuleParameters]:
        batch_norm = nn.BatchNorm2d(num_features)
        batch_norm.weight.data *= torch.randn_like(batch_norm.weight.data)

        batch_norm_mp = ModuleParameters(
            weight_a=batch_norm.weight,
            weight_b=batch_norm.weight,
            name="batch_norm",
            axis_to_permutation={0: Perm(torch.randperm(num_features))},
            bias_a=batch_norm.bias,
            bias_b=batch_norm.bias,
            module_type=nn.BatchNorm2d,
        )
        return batch_norm, batch_norm_mp

    @staticmethod
    def conv2d(
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int],
            bias: bool = False,
            padding: int = 0,
    ) -> tuple[nn.Conv2d, ModuleParameters]:
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, bias=bias, padding=padding
        )
        bias_ = conv2d.bias if bias else None
        assert isinstance(bias_, (nn.Parameter, type(None)))  # for mypy

        conv2d_mp = ModuleParameters(
            weight_a=conv2d.weight,
            weight_b=conv2d.weight,
            name="conv2d",
            axis_to_permutation={
                0: Perm(torch.randperm(out_channels)),
                1: Perm(torch.randperm(in_channels))
            },
            bias_a=bias_,
            bias_b=bias_,
            module_type=nn.Conv2d,
        )
        return conv2d, conv2d_mp

    def linear_path(self) -> tuple[nn.Module, list[list[ModuleParameters]]]:
        lin3_4, lin3_4_mp = self.linear(3, 4)
        lin4_4, lin4_4_mp = self.linear(4, 4)
        lin4_3, lin4_3_mp = self.linear(4, 3)
        ln4_one, ln4_one_mp = self.layernorm(4)
        ln4_two, ln4_two_mp = self.layernorm(4)
        ln3_one, ln3_mp_one = self.layernorm(3)
        ln3_two, ln3_mp_two = self.layernorm(3)

        model = nn.Sequential(
            ln3_one, lin3_4, self.relu,  # Start with a LayerNorm
            ln4_one, lin4_4, self.relu,  # LayerNorms in between
            ln4_two, lin4_3, self.relu,
            ln3_two  # End in a LayerNorm
        )
        seq = [
            ln3_mp_one, lin3_4_mp,
            ln4_one_mp, lin4_4_mp,
            ln4_two_mp, lin4_3_mp,
            ln3_mp_two
        ]
        return model, [seq]

    def residual_path_linear_simple(
            self
    ) -> tuple[nn.Module, list[list[ModuleParameters]]]:
        lin3_4, lin3_4_mp = self.linear(3, 4)
        lin4_4, lin4_4_mp = self.linear(4, 4)
        lin4_3, lin4_3_mp = self.linear(4, 3)
        ln4_one, ln4_one_mp = self.layernorm(4)
        ln4_two, ln4_two_mp = self.layernorm(4)
        ln3_one, ln3_mp_one = self.layernorm(3)
        ln3_two, ln3_mp_two = self.layernorm(3)
        relu = nn.ReLU()

        class ResBlockLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.path = nn.Sequential(
                    ln3_one, lin3_4, relu,  # Start with a LayerNorm
                    ln4_one, lin4_4, relu,  # LayerNorms in between
                    ln4_two, lin4_3, relu,
                    ln3_two  # End in a LayerNorm
                )

            def forward(self, input_tensor: torch.Tensor) -> Any:
                return self.path(input_tensor) + input_tensor

        model = ResBlockLinear()
        longpath = [
            ln3_mp_one, lin3_4_mp,
            ln4_one_mp, lin4_4_mp,
            ln4_two_mp, lin4_3_mp,
            ln3_mp_two
        ]
        shortpath: list[ModuleParameters] = []
        return model, [longpath, shortpath]

    def residual_path_linear_complex(
            self
    ) -> tuple[nn.Module, list[list[ModuleParameters]]]:
        lin3_4, lin3_4_mp = self.linear(3, 4)
        lin4_4, lin4_4_mp = self.linear(4, 4)
        lin4_3, lin4_3_mp = self.linear(4, 3)
        ln4_one, ln4_one_mp = self.layernorm(4)
        ln4_two, ln4_two_mp = self.layernorm(4)
        ln3_one, ln3_mp_one = self.layernorm(3)
        ln3_two, ln3_mp_two = self.layernorm(3)

        ln3_sc, ln3_sc_mp = self.layernorm(3)
        lin3_3_sc, lin3_3_sc_mp = self.linear(3, 3)
        relu = nn.ReLU()

        class ResBlockLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.path = nn.Sequential(
                    ln3_one, lin3_4, relu,  # Start with a LayerNorm
                    ln4_one, lin4_4, relu,  # LayerNorms in between
                    ln4_two, lin4_3, relu,
                    ln3_two  # End in a LayerNorm
                )
                self.shortcut = nn.Sequential(
                    ln3_sc, lin3_3_sc, relu
                )

            def forward(self, input_tensor: torch.Tensor) -> Any:
                return self.path(input_tensor) + self.shortcut(input_tensor)

        model = ResBlockLinear()

        longpath = [
            ln3_mp_one, lin3_4_mp,
            ln4_one_mp, lin4_4_mp,
            ln4_two_mp, lin4_3_mp,
            ln3_mp_two
        ]
        shortpath = [ln3_sc_mp, lin3_3_sc_mp]

        return model, [longpath, shortpath]

    def linear_path_conv2d_simple(
            self
    ) -> tuple[torch.Tensor, nn.Module, list[list[ModuleParameters]]]:
        x = torch.randn(2, 3, 36, 36)

        conv3_4, conv3_4_mp = self.conv2d(3, 4, (3, 3), padding=1)
        conv4_4, conv4_4_mp = self.conv2d(4, 4, (3, 3), padding=1)
        conv4_3, conv4_3_mp = self.conv2d(4, 3, (3, 3), padding=1)

        model = nn.Sequential(
            conv3_4, self.relu, conv4_4, self.relu, conv4_3, self.relu
        )
        path = [conv3_4_mp, conv4_4_mp, conv4_3_mp]

        return x, model, [path]

    def linear_path_conv2d_medium(
            self
    ) -> tuple[torch.Tensor, nn.Module, list[list[ModuleParameters]]]:
        x = torch.randn(2, 3, 36, 36)

        conv3_4, conv3_4_mp = self.conv2d(3, 4, (3, 3), padding=1)
        conv4_4, conv4_4_mp = self.conv2d(4, 4, (3, 3), padding=1)
        conv4_3, conv4_3_mp = self.conv2d(4, 3, (3, 3), padding=1)

        ln1, ln1_mp = self.layernorm(conv3_4(x).shape)
        ln2, ln2_mp = self.layernorm(conv4_4(conv3_4(x)).shape)

        model = nn.Sequential(
            conv3_4, self.relu, ln1, conv4_4, self.relu, ln2, conv4_3, self.relu
        )
        path = [conv3_4_mp, ln1_mp, conv4_4_mp, ln2_mp, conv4_3_mp]

        return x, model, [path]

    def linear_path_conv2d_complex(
            self
    ) -> tuple[torch.Tensor, nn.Module, list[list[ModuleParameters]]]:
        x = torch.randn(2, 3, 36, 36)

        conv3_4, conv3_4_mp = self.conv2d(3, 4, (3, 3), padding=1)
        conv4_4, conv4_4_mp = self.conv2d(4, 4, (3, 3), padding=1)
        conv4_3, conv4_3_mp = self.conv2d(4, 3, (3, 3), padding=1)

        ln0, ln0_mp = self.layernorm(x.shape)
        ln1, ln1_mp = self.layernorm(conv3_4(x).shape)
        ln2, ln2_mp = self.layernorm(conv4_4(conv3_4(x)).shape)
        ln3, ln3_mp = self.layernorm(conv4_3(conv4_4(conv3_4(x))).shape)

        model = nn.Sequential(
            ln0, conv3_4, self.relu,
            ln1, conv4_4, self.relu,
            ln2, conv4_3, self.relu,
            ln3
        )
        path = [
            ln0_mp, conv3_4_mp,
            ln1_mp, conv4_4_mp,
            ln2_mp, conv4_3_mp,
            ln3_mp
        ]
        return x, model, [path]

    def residual_path_conv2d_simple(
            self
    ) -> tuple[torch.Tensor, nn.Module, list[list[ModuleParameters]]]:
        x = torch.randn(2, 3, 36, 36)

        conv3_4, conv3_4_mp = self.conv2d(3, 4, (3, 3), padding=1)
        conv4_4, conv4_4_mp = self.conv2d(4, 4, (3, 3), padding=1)
        conv4_3, conv4_3_mp = self.conv2d(4, 3, (3, 3), padding=1)

        ln0, ln0_mp = self.layernorm(x.shape)
        ln1, ln1_mp = self.layernorm(conv3_4(x).shape)
        ln2, ln2_mp = self.layernorm(conv4_4(conv3_4(x)).shape)
        ln3, ln3_mp = self.layernorm(conv4_3(conv4_4(conv3_4(x))).shape)
        relu = nn.ReLU()

        class ResBlockConv2d(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.path = nn.Sequential(
                    ln0, conv3_4, relu,  # Start with a LayerNorm
                    ln1, conv4_4, relu,  # LayerNorms in between
                    ln2, conv4_3, relu,
                    ln3  # End in a LayerNorm
                )

            def forward(self, input_tensor: torch.Tensor) -> Any:
                return self.path(input_tensor) + input_tensor

        model = ResBlockConv2d()
        longpath = [
            ln0_mp, conv3_4_mp,
            ln1_mp, conv4_4_mp,
            ln2_mp, conv4_3_mp,
            ln3_mp
        ]
        shortpath: list[ModuleParameters] = []
        return x, model, [longpath, shortpath]

    def linear_path_with_batch_norm(
            self
    ) -> tuple[torch.Tensor, nn.Module, list[list[ModuleParameters]]]:
        x = torch.randn(2, 3, 36, 36)

        conv3_4, conv3_4_mp = self.conv2d(3, 4, (3, 3))
        conv4_4, conv4_4_mp = self.conv2d(4, 4, (3, 3))
        conv4_3, conv4_3_mp = self.conv2d(4, 3, (3, 3))

        bn3, bn3_mp = self.batchnorm2d(3)
        bn4, bn4_mp = self.batchnorm2d(4)
        bn4_2, bn4_2_mp = self.batchnorm2d(4)
        bn3_2, bn3_2_mp = self.batchnorm2d(3)

        model = nn.Sequential(
            bn3, conv3_4, self.relu,
            bn4, conv4_4, self.relu,
            bn4_2, conv4_3, self.relu,
            bn3_2
        )

        path = [
            bn3_mp, conv3_4_mp,
            bn4_mp, conv4_4_mp,
            bn4_2_mp, conv4_3_mp,
            bn3_2_mp
        ]

        return x, model, [path]
