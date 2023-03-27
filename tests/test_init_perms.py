from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18  # type: ignore[import]

from rebasin.perm.init_perms import PermutationInitializer
from rebasin.perm.structs import AppliesTo

from .fixtures.models import MLP


def test_init_permutations_mlp() -> None:
    """Test the initialization of the permutations."""
    model_a = MLP(5)
    model_b = MLP(5)
    input_data = torch.randn(5)
    perm_init = PermutationInitializer(model_a, model_b, input_data)

    common_init_tests(model_b, perm_init)


def test_init_permutations_resnet() -> None:
    """Test the initialization of the permutations."""
    model_a = resnet18()
    model_b = resnet18()
    input_data = torch.randn(1, 3, 224, 224)
    perm_init = PermutationInitializer(model_a, model_b, input_data)

    common_init_tests(model_b, perm_init)


def common_init_tests(model_b: nn.Module, perm_init: PermutationInitializer) -> None:
    modules = [
        m for m in model_b.modules() if hasattr(m, "weight") and m.weight is not None
    ]

    assert len(perm_init.permutations_init) > len(modules)  # several axes per module

    for permutation in perm_init.permutations_init:
        for module_info in permutation.modules:
            assert module_info.module_b in model_b.modules()
            pshape = permutation.perm_indices.shape

            if module_info.applies_to in (AppliesTo.WEIGHT, AppliesTo.BOTH):
                wshape = module_info.module_b.weight.shape
                assert isinstance(wshape, torch.Size)
                assert isinstance(pshape, torch.Size)
                assert wshape[module_info.axis] == pshape[0]
            elif module_info.applies_to in (AppliesTo.BIAS, AppliesTo.BOTH):
                bshape = module_info.module_b.bias.shape
                assert isinstance(bshape, torch.Size)
                assert isinstance(pshape, torch.Size)
                assert bshape[0] == pshape[0]

    for permutation in perm_init.permutations:
        assert all(mi.module_b in model_b.modules() for mi in permutation.modules)

    # As several of the modules in the tested models are compatible,
    #   some permutations should be merged.
    assert len(perm_init.permutations) < len(perm_init.permutations_init)

    # Bias isn't turned off in either resnet18 or MLP
    #   -> applies_to == BOTH should be present.
    if any(hasattr(m, "bias") and m.bias is not None for m in modules):
        assert any(
            mi.applies_to == AppliesTo.BOTH
            for p in perm_init.permutations
            for mi in p.modules
        )
