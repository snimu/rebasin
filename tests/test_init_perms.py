from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18  # type: ignore[import]

from rebasin import util
from rebasin.weight_matching.init_perms import PermutationInitializer
from rebasin.weight_matching.structs import AxisType

from .fixtures.models import MLP, ModuleWithWeirdWeightAndBiasNames


def test_init_permutations_weird_weight_bias_names() -> None:
    model_a = ModuleWithWeirdWeightAndBiasNames()
    model_b = ModuleWithWeirdWeightAndBiasNames()
    x = torch.randn(15)
    perm_init = PermutationInitializer(model_a, model_b, (x,))

    for perm in perm_init.permutations:
        names = [p.name for p in perm.parameters]
        if perm.parameters[0].name == "xyzweight":
            assert len(perm.parameters) == 1  # should not be associated with xyzbias

        if (
                perm.parameters[0].axis == 0
                and perm.parameters[0].name not in ("xyzweight", "xyzbias")
        ):
            assert len(perm.parameters) == 2  # every weight has a bias except xyz...
            if "weightabc" in names:
                assert "abcbias" in names
            if "defweightghi" in names:
                assert "defbiasghi" in names
            if "jklweight" in names:
                assert "biasjkl" in names


def test_init_permutations_multihead_attention() -> None:
    """Test the initialization of the permutations."""
    model_a = nn.MultiheadAttention(8, 8)
    model_b = nn.MultiheadAttention(8, 8)
    x = torch.randn(8, 8)
    perm_init = PermutationInitializer(model_a, model_b, (x, x, x))

    for permutation in perm_init.permutations:
        parameter_names = [p.name for p in permutation.parameters]
        for param_info in permutation.parameters:
            # Bias only on axis 0
            if "in_proj_weight" in parameter_names and param_info.axis == 0:
                assert "in_proj_bias" in parameter_names
                assert "out_proj.weight" not in parameter_names
            if "out_proj.weight" in parameter_names and param_info.axis == 0:
                assert "out_proj.bias" in parameter_names
                assert "in_proj_weight" not in parameter_names


def test_init_permutations_mlp() -> None:
    """Test the initialization of the permutations."""
    model_a = MLP(5)
    model_b = MLP(5)
    input_data = torch.randn(5)
    perm_init = PermutationInitializer(model_a, model_b, input_data)

    common_init_tests(model_a, model_b, perm_init)


def test_init_permutations_resnet() -> None:
    """Test the initialization of the permutations."""
    model_a = resnet18()
    model_b = resnet18()
    input_data = torch.randn(1, 3, 224, 224)
    perm_init = PermutationInitializer(model_a, model_b, input_data)

    common_init_tests(model_a, model_b, perm_init)


def common_init_tests(
        model_a: nn.Module, model_b: nn.Module, perm_init: PermutationInitializer
) -> None:
    modules = [
        m for m in model_b.modules()
    ]
    module_ids = [id(m) for m in modules]

    assert len(perm_init.permutations_init) > len(modules)  # several axes per module

    for permutation in perm_init.permutations:
        for param_info in permutation.parameters:
            assert param_info.module_id in perm_init.id_to_permutation_init
            assert param_info.module_id in module_ids
            assert util.contains_parameter(model_a.parameters(), param_info.param_a)
            assert util.contains_parameter(model_b.parameters(), param_info.param_b)

            assert permutation.perm_indices.shape[0] == \
                   param_info.param_a.shape[param_info.axis]

            assert permutation.perm_indices.shape[0] == \
                   param_info.param_b.shape[param_info.axis]

    # As several of the modules in the tested models are compatible,
    #   some permutations should be merged.
    assert len(perm_init.permutations) < len(perm_init.permutations_init)

    # Bias isn't turned off in either resnet18 or MLP
    #   -> axis_type = AxisType.NEITHER should be present.
    has_bias = any(
        "bias" in name
        for m in modules
        for name, _ in m.named_parameters()
    )

    if has_bias:
        assert any(
            param_info.axis_type == AxisType.NEITHER
            for permutation in perm_init.permutations
            for param_info in permutation.parameters
        )
