from __future__ import annotations

import torch
from torch import nn

from rebasin.initialization._permutation import ParameterInfo, Permutation


class PermutationInitializer:
    def __init__(self, model_a: nn.Module, model_b: nn.Module) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self._perm_len_to_perm: dict[int, Permutation] = {}
        self.permutations: list[Permutation] = []
        self.param_to_param_infos: dict[nn.Parameter, list[ParameterInfo]] = {}
        self._init_permutations()

    def _init_permutations(self) -> None:
        """Initialize the permutations."""
        for (name_a, parameter_a), (name_b, parameter_b) in zip(  # noqa: B905
                self.model_a.named_parameters(),
                self.model_b.named_parameters(),
        ):
            self._check_model_similarity(name_a, name_b, parameter_a, parameter_b)
            param_infos = self._get_param_infos(name_a, parameter_a, parameter_b)
            for param_info in param_infos:
                ax_len = param_info.param_a.shape[param_info.axis]
                if ax_len not in self._perm_len_to_perm:
                    perm = Permutation(param_infos=[param_info])
                    self._perm_len_to_perm[ax_len] = perm
                else:
                    self._perm_len_to_perm[ax_len].param_infos.append(param_info)

                if parameter_b not in self.param_to_param_infos:
                    self.param_to_param_infos[parameter_b] = [param_info]
                else:
                    self.param_to_param_infos[parameter_b].append(param_info)

        for permutation in self._perm_len_to_perm.values():
            self.permutations.append(permutation)

    @staticmethod
    def _check_model_similarity(
            name_a: str,
            name_b: str,
            parameter_a: nn.Parameter,
            parameter_b: nn.Parameter
    ) -> None:
        """Check that the models are similar."""
        if name_a != name_b:
            raise ValueError(
                f"Model parameters do not match: {name_a} != {name_b}"
            )
        if parameter_a.shape != parameter_b.shape:
            raise ValueError(
                f"Model parameters do not match: "
                f"{parameter_a.shape} != {parameter_b.shape}"
            )
        if parameter_a.dtype != parameter_b.dtype:
            raise ValueError(
                f"Model parameters do not match: "
                f"{parameter_a.dtype} != {parameter_b.dtype}"
            )

    @staticmethod
    def _get_param_infos(
            name: str,
            param_a: nn.Parameter,
            param_b: nn.Parameter,
    ) -> list[ParameterInfo]:
        if not ("weight" in name or "bias" in name):
            return []

        if not isinstance(param_a, (torch.Tensor, nn.Parameter)):
            return []
        if not isinstance(param_b, (torch.Tensor, nn.Parameter)):
            return []

        assert isinstance(param_a.shape, torch.Size)
        assert isinstance(param_b.shape, torch.Size)

        if len(param_b.shape) == 0:
            return []  # cannot permute scalars

        parameters = []

        if "weight" in name:
            parameters.append(
                ParameterInfo(
                    name=name,
                    param_a=param_a,
                    param_b=param_b,
                    axis=0,
                )
            )
            # Only for nd-weights, where n > 1 (for example LayerNorm)
            if len(param_b.shape) > 1:
                parameters.append(
                    ParameterInfo(
                        name=name,
                        param_a=param_a,
                        param_b=param_b,
                        axis=1,
                    )
                )
        elif "bias" in name:
            parameters.append(
                ParameterInfo(
                    name=name,
                    param_a=param_a,
                    param_b=param_b,
                    axis=0,
                    # This is not associated with other parameters:
                )
            )

        return parameters
