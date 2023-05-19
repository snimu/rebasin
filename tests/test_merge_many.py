from __future__ import annotations

import sys

import pytest
import torch

from rebasin import MergeMany
from tests.fixtures.models import MLP


def test_merge_many__get_mean_model() -> None:
    models = [MLP(10, 10), MLP(10, 10), MLP(10, 10)]
    x = torch.randn(10)
    merge_many = MergeMany(models=models, working_model=MLP(10, 10), input_data=x)
    mean_model = merge_many._get_mean_model()

    manual_mean_model = MLP(10, 10)
    manual_mean_model.load_state_dict(models[0].state_dict())
    for model in models[1:]:
        for param, mean_param in zip(
                model.parameters(), manual_mean_model.parameters()
        ):
            mean_param.data.add_(param.data)

    for param in manual_mean_model.parameters():
        param.data.div_(3.0)

    for param, mean_param in zip(
            mean_model.parameters(), manual_mean_model.parameters()
    ):
        assert torch.allclose(param, mean_param)

    # Test with except_index.
    mean_model = merge_many._get_mean_model(except_index=0)
    manual_mean_model = MLP(10, 10)
    manual_mean_model.load_state_dict(models[1].state_dict())
    for param, mean_param in zip(
            models[2].parameters(), manual_mean_model.parameters()
    ):
        mean_param.data.add_(param.data)

    for param in manual_mean_model.parameters():
        param.data.div_(2.0)

    for param, mean_param in zip(
            mean_model.parameters(), manual_mean_model.parameters()
    ):
        assert torch.allclose(param, mean_param)


def test_merge_many() -> None:
    models = [MLP(5, 5), MLP(5, 5), MLP(5, 5)]
    x = torch.randn(5)
    merge_many = MergeMany(models=models, working_model=MLP(5, 5), input_data=x)

    # I cannot write any proper tests here, because I would need data for that,
    #   and I don't want to download any for testing.
    # I will just test that the code runs without errors.
    merged_model = merge_many.run()


@pytest.mark.skipif(
    "--full-suite" not in sys.argv,
    reason="This test takes a long time to run."
)
def test_merge_many_large_mlp() -> None:
    models = [MLP(50, 50), MLP(50, 50), MLP(50, 50)]
    x = torch.randn(50)
    merge_many = MergeMany(models=models, working_model=MLP(50, 50), input_data=x)

    # I cannot write any proper tests here, because I would need data for that,
    #   and I don't want to download any for testing.
    # I will just test that the code runs without errors.
    _ = merge_many.run()
