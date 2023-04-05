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
    return ((x - y) ** 2).sum()  # type: ignore[no-any-return]


def test_loss_fn() -> None:
    """Test that loss_fn is working as expected."""
    l1 = loss_fn(torch.tensor([1.0]), torch.tensor([2.0]))
    l2 = loss_fn(torch.tensor([1.0]), torch.tensor([1.0]))
    assert l1 > l2


class BaseClass:
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

    @property
    def eval_fn(self) -> Any:
        dl = self.val_dl

        def eval_fn(model: nn.Module, device: torch.device | str | None) -> float:
            loss = 0.0
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                loss += self.loss_fn(model(x), y)
            return loss / len(dl)

        return eval_fn


class TestInterpolation(BaseClass):
    """Test sanity checks of Interpolation class."""

    def test_sanity_checks_model(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=[],
                eval_fn=self.eval_fn,
                eval_mode="min",
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=MLP(5),  # type: ignore[arg-type]
                eval_fn=self.eval_fn,
                eval_mode="min",
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=[*self.models, "not a model"],  # type: ignore[list-item]
                eval_fn=self.eval_fn,
                eval_mode="min",
            )

    def test_sanity_checks_eval_fn(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                eval_fn=0,
                eval_mode="min",
            )

    def test_sanity_checks_dataloader(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                eval_fn=self.eval_fn,
                eval_mode="min",
                train_dataloader=0,  # type: ignore[arg-type]
            )

    def test_sanity_checks_devices(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                eval_fn=self.eval_fn,
                eval_mode="min",
                devices=["cpu", "cpu", "cpu"],
                device_interp=None
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                eval_fn=self.eval_fn,
                eval_mode="min",
                devices=["cpu", "cpu", "cpu", "cpu"],  # too long
                device_interp="cpu"
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                eval_fn=self.eval_fn,
                eval_mode="min",
                devices=None,
                device_interp="cpu"
            )

    def test_sanity_checks_input_indices(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                eval_fn=self.eval_fn,
                eval_mode="min",
                input_indices=0.5  # type: ignore[arg-type]
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                eval_fn=self.eval_fn,
                eval_mode="min",
                input_indices=[0.5]  # type: ignore[list-item]
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                eval_fn=self.eval_fn,
                eval_mode="min",
                input_indices=["not", "an", "int"]  # type: ignore[list-item]
            )

    def test_sanity_checks_save_vars(self) -> None:
        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                eval_fn=self.eval_fn,
                eval_mode="min",
                savedir=0  # type: ignore[arg-type]
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                eval_fn=self.eval_fn,
                eval_mode="min",
                save_all="not a bool"  # type: ignore[arg-type]
            )

        with pytest.raises(AssertionError):
            interp.Interpolation(
                models=self.models,
                eval_fn=self.eval_fn,
                eval_mode="min",
                save_all=True,
                savedir=None
            )


class TestLerpSimple(BaseClass):
    """Test the LerpSimple class."""
    @property
    def mlps(self) -> list[nn.Module]:
        return [MLP(5), MLP(5)]

    def test_settings_are_plausible(self) -> None:
        model_a, model_b = self.mlps

        # Must be different models
        for p1, p2 in zip(model_a.parameters(), model_b.parameters(), strict=True):
            assert not torch.allclose(p1, p2)

        # Must produce different outputs
        x = torch.rand(5)
        assert not torch.allclose(model_a(x), model_b(x))

        # Must have different losses
        val_dl = self.val_dl
        x, y = next(iter(val_dl))
        loss_a = loss_fn(model_a(x), y)
        loss_b = loss_fn(model_b(x), y)

        assert loss_a != loss_b

    def test_mlp(self) -> None:
        lerp = interp.LerpSimple(
            models=self.mlps,
            eval_fn=self.eval_fn,
            eval_mode="min",
        )
        lerp.interpolate(steps=10)

        assert len(lerp.metrics_interpolated) == 10
        assert len(lerp.metrics_models) == 2

        # With random data, there should be no duplicate losses.
        assert len(set(lerp.metrics_models)) == 2
        assert len(set(lerp.metrics_interpolated)) == 10

        # Check that the best model is the one with the lowest loss
        best_model_loss = min(lerp.metrics_models)
        best_interp_loss = min(lerp.metrics_interpolated)
        best_loss = min(best_model_loss, best_interp_loss)
        assert abs(best_loss - lerp.best_metric) < 1e-6 * lerp.best_metric


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU tests")
class TestInterpolationGPU(BaseClass):
    def test_mlp(self) -> None:
        models = [MLP(5).to("cuda"), MLP(5)]
        devices = ["cuda", "cpu"]
        lerp = interp.LerpSimple(
            models=models,
            eval_fn=self.eval_fn,
            eval_mode="min",
            devices=devices,
            device_interp="cuda",
        )
        lerp.interpolate(steps=10)
        assert len(lerp.metrics_interpolated) == 10
        assert len(lerp.metrics_models) == 2

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs")
    def test_mlp_multi_gpu(self) -> None:
        """Test whether it works with :code:`device_interp not in devices`."""

        models = [MLP(5).to("cuda:0"), MLP(5).to("cuda:1")]
        devices = ["cuda:0", "cuda:1"]
        lerp = interp.LerpSimple(
            models=models,
            eval_fn=self.eval_fn,
            eval_mode="min",
            devices=devices,
            device_interp="cuda",
        )
        lerp.interpolate(steps=10)
        assert len(lerp.metrics_interpolated) == 10
        assert len(lerp.metrics_models) == 2
