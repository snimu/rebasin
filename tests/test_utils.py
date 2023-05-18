from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rebasin.utils import contains_parameter, get_inputs_labels, recalculate_batch_norms
from tests.fixtures.models import SaveCallCount


def test_recalculate_batch_norms() -> None:
    conv = nn.Conv2d(1, 1, 3)
    bn = nn.BatchNorm2d(1)
    scc = SaveCallCount()
    model = torch.nn.Sequential(conv, bn, scc)
    tensors = [torch.rand(10, 1, 10, 10) for _ in range(40)]  # 20 times x, y
    dataset = TensorDataset(*tensors)
    dataloader = DataLoader(dataset, batch_size=5)

    # Test whether recalculate_batch_norms works
    assert isinstance(bn.running_mean, torch.Tensor)
    assert torch.all(bn.running_mean == torch.zeros_like(bn.running_mean))
    recalculate_batch_norms(model, dataloader, [0], None, False)
    assert torch.all(bn.running_mean != torch.zeros_like(bn.running_mean))
    assert scc.call_count == len(dataloader)  # Test that scc works

    # Test whether recalculate_batch_norms works in eval mode and resets it afterward
    model.eval()
    bn.running_mean = torch.zeros_like(bn.running_mean)  # reset running_mean
    recalculate_batch_norms(model, dataloader, [0], None, False)
    assert model.training is False  # Reset to eval mode
    assert torch.all(bn.running_mean != torch.zeros_like(bn.running_mean))  # Did work

    # Test whether recalculate_batch_norms stops early
    #   if there are no BatchNorms in the model
    scc.call_count = 0
    model = torch.nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3), scc)
    recalculate_batch_norms(model, dataloader, 0, None, False)
    assert scc.call_count == 0

    # Test whether the dataset_percentage-option works
    scc.call_count = 0
    model = torch.nn.Sequential(conv, bn, scc)
    recalculate_batch_norms(model, dataloader, [0], None, False, dataset_percentage=0.5)
    assert scc.call_count == int(len(dataloader) * 0.5 + 1e-6)


def test_get_inputs_labels() -> None:
    # Setup
    in1, in2, in3 = torch.rand(10, 3), torch.rand(10, 3), torch.rand(10, 3)
    out1, out2, out3 = torch.rand(10, 3), torch.rand(10, 3), torch.rand(10, 3)

    # For mypy:
    inputs: list[Any]
    labels: list[Any]
    batch: tuple[torch.Tensor, ...]

    # Test default indices
    batch = in1, out1
    inputs, labels = get_inputs_labels(batch)
    assert torch.allclose(inputs[0], in1)
    assert torch.allclose(labels[0], out1)

    # Test custom indices
    batch = in1, in2, out1, out2
    inputs, labels = get_inputs_labels(
        batch, input_indices=[0, 1], label_indices=[2, 3]
    )
    assert torch.allclose(inputs[0], in1)
    assert torch.allclose(inputs[1], in2)
    assert torch.allclose(labels[0], out1)
    assert torch.allclose(labels[1], out2)

    # Test mixed indices
    batch = in2, in3, in1, out2, out3, out1
    inputs, labels = get_inputs_labels(
        batch, input_indices=[2, 0, 1], label_indices=[5, 3, 4]
    )
    assert torch.allclose(inputs[0], in1)
    assert torch.allclose(inputs[1], in2)
    assert torch.allclose(inputs[2], in3)
    assert torch.allclose(labels[0], out1)
    assert torch.allclose(labels[1], out2)
    assert torch.allclose(labels[2], out3)


def test_contains_parameter() -> None:
    p1 = nn.Parameter(torch.rand(3, 3))
    p2 = nn.Parameter(torch.rand(3))
    p3 = nn.Parameter(torch.rand(5))

    assert contains_parameter((p1, p2), p1)
    assert contains_parameter((p1, p2), p2)
    assert not contains_parameter((p1, p2), p3)
