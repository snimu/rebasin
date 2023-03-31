from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from rebasin import interpolation as interp
from tests.fixtures.datasets import RandomDataset
from tests.fixtures.models import MLP


def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - y) ** 2)


class TestInterpolation:
    """Test sanity checks of Interpolation class."""
    @property
    def models(self) -> list[nn.Module]:
        return [MLP(5), MLP(5)]

    @property
    def train_dataloader(
            self
    ) -> torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(RandomDataset(shape=(5,), length=10))

    @property
    def val_dataloader(
            self
    ) -> torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(RandomDataset(shape=(5,), length=4))

    def test_sanity_checks_model(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=[],
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=MLP(5),  # type: ignore[arg-type]
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                input_indices=1
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=[*self.models, "not a model"],  # type: ignore[list-item]
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                input_indices=1
            )

    def test_sanity_checks_loss_fn(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=0,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader
            )

    def test_sanity_checks_dataloaders(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=loss_fn,
                train_dataloader=0,  # type: ignore[arg-type]
                val_dataloader=self.val_dataloader
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=0  # type: ignore[arg-type]
            )

    def test_sanity_checks_input_indices(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                input_indices=0.5  # type: ignore[arg-type]
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                input_indices=[0.5]  # type: ignore[list-item]
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                input_indices=["not", "an", "int"]  # type: ignore[list-item]
            )

    def test_sanity_checks_output_indices(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                output_indices=0.5  # type: ignore[arg-type]
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                output_indices=[0.5]  # type: ignore[list-item]
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                output_indices=["not", "an", "int"]  # type: ignore[list-item]
            )

    def test_sanity_checks_save_vars(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                savedir=0  # type: ignore[arg-type]
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                save_all="not a bool"  # type: ignore[arg-type]
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                loss_fn=loss_fn,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                save_all=True,
                savedir=None
            )
