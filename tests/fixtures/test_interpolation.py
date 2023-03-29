from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from rebasin.interpolation import LerpSimple
from tests.fixtures.datasets import RandomDataset
from tests.fixtures.models import MLP


class TestLerpSimple:

    @staticmethod
    def loss_fn(x: torch.Tensor, y: torch.Tensor) -> float:
        return float(((x - y) ** 2).sum())

    @property
    def models(self) -> list[MLP]:
        return [MLP(5) for _ in range(3)]

    @property
    def train_dl(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(RandomDataset((5,), 5), batch_size=1, shuffle=True)

    @property
    def val_dl(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(RandomDataset((5,), 2), batch_size=1, shuffle=False)

    def test_mlp(self) -> None:
        lerp = LerpSimple(
            models=self.models,
            loss_fn=self.loss_fn,
            train_dataloader=self.train_dl,
            val_dataloader=self.val_dl,
        )
        lerp.interpolate()

        # Check that the losses are assigned correctly
        loss = float(lerp.evaluate_model(lerp.best_model))
        assert abs(loss - lerp.best_loss) < 1e-6 * lerp.best_loss

        # Check that the best model is the one with the lowest loss
        best_model_loss = min(lerp.losses_original)
        best_interp_loss = min(lerp.losses_interpolated)
        best_loss = min(best_model_loss, best_interp_loss)
        assert abs(best_loss - lerp.best_loss) < 1e-6 * lerp.best_loss

    def test_sanity_checks(self) -> None:
        with pytest.raises(AssertionError):
            LerpSimple(
                models=1,  # type: ignore[arg-type]
                loss_fn=self.loss_fn,
                train_dataloader=self.train_dl,
                val_dataloader=self.val_dl,
                save_all=False,
                savedir="test",
            )

        with pytest.raises(AssertionError):
            LerpSimple(
                models=[1, 2, 3],  # type: ignore[list-item]
                loss_fn=self.loss_fn,
                train_dataloader=self.train_dl,
                val_dataloader=self.val_dl,
                save_all=False,
                savedir="test",
            )

        with pytest.raises(AssertionError):
            LerpSimple(
                models=self.models,
                loss_fn=1,
                train_dataloader=self.train_dl,
                val_dataloader=self.val_dl,
                save_all=False,
                savedir="test",
            )

        with pytest.raises(AssertionError):
            LerpSimple(
                models=self.models,
                loss_fn=self.loss_fn,
                train_dataloader=1,  # type: ignore[arg-type]
                val_dataloader=self.val_dl,
                save_all=False,
                savedir="test",
            )

        with pytest.raises(AssertionError):
            LerpSimple(
                models=self.models,
                loss_fn=self.loss_fn,
                train_dataloader=self.train_dl,
                val_dataloader=1,  # type: ignore[arg-type]
                save_all=True,
                savedir="test",
            )

        with pytest.raises(AssertionError):
            LerpSimple(
                models=self.models,
                loss_fn=self.loss_fn,
                train_dataloader=self.train_dl,
                val_dataloader=self.val_dl,
                save_all="False",  # type: ignore[arg-type]
                savedir="test",
            )

        with pytest.raises(AssertionError):
            LerpSimple(
                models=self.models,
                loss_fn=self.loss_fn,
                train_dataloader=self.train_dl,
                val_dataloader=self.val_dl,
                save_all=False,
                savedir=1,  # type: ignore[arg-type]
            )

        with pytest.raises(AssertionError):
            LerpSimple(
                models=self.models,
                loss_fn=self.loss_fn,
                train_dataloader=self.train_dl,
                val_dataloader=self.val_dl,
                save_all=True,
                savedir=None,
            )

