from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18  # type: ignore[import]

from rebasin.perm.init_perms import PermutationInitializer
from rebasin.perm.structs import AppliesTo

from .fixtures.models import MLP


def test_init_permutations_mlp() -> None:
    """Test the initialization of the permutations."""
    model = MLP(5)
    input_data = torch.randn(5)
    perm_init = PermutationInitializer(model, input_data)

    common_init_tests(model, perm_init)


def test_init_permutations_resnet() -> None:
    """Test the initialization of the permutations."""
    model = resnet18()
    input_data = torch.randn(1, 3, 224, 224)
    perm_init = PermutationInitializer(model, input_data)

    common_init_tests(model, perm_init)


def common_init_tests(model: nn.Module, perm_init: PermutationInitializer) -> None:
    for permutation in perm_init.permutations:
        assert permutation.module in model.modules()

        if permutation.applies_to in (AppliesTo.WEIGHT, AppliesTo.BOTH):
            wshape = permutation.module.weight.shape
            assert isinstance(wshape, torch.Size)
            pshape = permutation.perm_indices.shape
            assert isinstance(pshape, torch.Size)

            assert wshape[permutation.axis] == pshape[0]
        elif permutation.applies_to in (AppliesTo.BIAS, AppliesTo.BOTH):
            bshape = permutation.module.bias.shape
            assert isinstance(bshape, torch.Size)
            pshape = permutation.perm_indices.shape
            assert isinstance(pshape, torch.Size)

            assert bshape[0] == pshape[0]

    for id_, permutations in perm_init.id_to_permutation.items():
        assert all(p.module in model.modules() for p in permutations)
        assert all(p.node.compute_unit_id == id_ for p in permutations)
