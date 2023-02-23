from __future__ import annotations

import torch


def _init_identity(tensor: torch.Tensor, n_eye: int) -> torch.Tensor:
    if len(tensor.size()) == 2:
        return torch.eye(n_eye)

    for i, tensor_ in enumerate(tensor):
        tensor[i] = _init_identity(tensor_, n_eye)
    return tensor


def identity_tensor(tensor: torch.Tensor) -> torch.Tensor:
    dims = list(tensor.size())
    assert len(dims) >= 2, "Currently only works for tensors of dimension >= 2."
    n_eye = max(dims[-1], dims[-2])
    dims[-2], dims[-1] = n_eye, n_eye

    perm = torch.empty(dims)
    perm = _init_identity(perm, n_eye)
    return perm
