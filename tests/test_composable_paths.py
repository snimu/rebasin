from __future__ import annotations

import torch
from torchview import draw_graph

from rebasin import ComposablePaths

from .fixtures.models import MLP


def test_get_edge_list() -> None:
    mlp = MLP(5)
    input_data = torch.ones(5)
    modules = list(mlp.modules())
    edge_list = draw_graph(mlp, input_data=input_data, depth=100).edge_list
    new_edge_list = ComposablePaths(mlp, input_data).edge_list

    for edge_old, edge_new in zip(edge_list, new_edge_list, strict=True):
        lo, ro = edge_old
        ln, rn = edge_new

        if isinstance(ln, torch.nn.Module):
            assert ln in modules
        else:
            assert isinstance(ln, type(lo))

        if isinstance(rn, torch.nn.Module):
            assert rn in modules
        else:
            assert isinstance(rn, type(ro))
