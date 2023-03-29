from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rebasin import recalculate_batch_norms
from tests.fixtures.models import SaveCallCount


def test_recalculate_batch_norms() -> None:
    conv = nn.Conv2d(1, 1, 3)
    bn = nn.BatchNorm2d(1)
    scc = SaveCallCount()
    model = torch.nn.Sequential(conv, bn, scc)
    dataloader = DataLoader(TensorDataset(torch.rand(10, 1, 10, 10)), batch_size=10)

    # Test whether recalculate_batch_norms works
    assert isinstance(bn.running_mean, torch.Tensor)
    assert torch.all(bn.running_mean == torch.zeros_like(bn.running_mean))
    recalculate_batch_norms(model, dataloader, [0])
    assert torch.all(bn.running_mean != torch.zeros_like(bn.running_mean))
    assert scc.call_count > 0  # Test that scc works

    # Test whether recalculate_batch_norms works in eval mode and resets it afterwards
    model.eval()
    bn.running_mean = torch.zeros_like(bn.running_mean)  # reset running_mean
    recalculate_batch_norms(model, dataloader, [0])
    assert model.training is False  # Reset to eval mode
    assert torch.all(bn.running_mean != torch.zeros_like(bn.running_mean))  # Did work

    # Test whether recalculate_batch_norms stops early
    #   if there are no BatchNorms in the model
    scc.call_count = 0
    model = torch.nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3), scc)
    recalculate_batch_norms(model, dataloader, [0])
    assert scc.call_count == 0
