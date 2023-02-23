from __future__ import annotations

import torch

from rebasin.math import identity_tensor


def test_identity_tensor() -> None:
    weights = (
        torch.randn(2, 3, 4),
        torch.randn(1, 2, 3, 4),
        torch.randn(3, 3, 3),
        torch.randn(2, 2)
    )

    for weight in weights:
        identity = identity_tensor(weight)
        assert torch.all(weight == weight @ identity)
