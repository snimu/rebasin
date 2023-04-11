"""
Apply rebasin to real models.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10  # type: ignore[import]

from rebasin import PermutationCoordinateDescent
from rebasin.interpolation import LerpSimple
from rebasin.util import recalculate_batch_norms
from tests.fixtures.mandw import MODEL_NAMES, MODELS_AND_WEIGHTS


class TorchvisionEval:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--models", type=str, nargs='*')
        parser.add_argument("-a", "--all", action="store_true", default=False)
        parser.add_argument("-v", "--verbose", action="store_true", default=True)
        parser.add_argument("-i", "--ignore_bn", action="store_true", default=False)
        parser.add_argument("-b", "--batch_size", type=int, default=64)
        parser.add_argument(
            "-p", "--percent_eval",
            type=float, default=100,
            help="Percent of data to evaluate on. Between 0 and 100."
        )

        self.hparams = parser.parse_args()

        if self.hparams.models is not None and not self.hparams.all:
            assert self.hparams.models, "Must specify models or all"

        if self.hparams.models is None:
            self.hparams.models = []
            assert self.hparams.all, "Must specify models or all"

        if self.hparams.all:
            self.hparams.models = MODEL_NAMES

        for model_name in self.hparams.models:
            assert model_name in MODEL_NAMES, f"{model_name} not in MODEL_NAMES"

        assert self.hparams.batch_size > 0, "Batch size must be greater than 0"
        assert 0 < self.hparams.percent_eval <= 100, "Percent eval must be in ]0, 100]"
        self.hparams.percent_eval = self.hparams.percent_eval / 100

        self.root_dir = os.path.join(os.path.dirname(Path(__file__)), "data")
        self.results_dir = os.path.join(os.path.dirname(Path(__file__)), "results")

        self.train_dl_a = DataLoader(  # Download the data
            CIFAR10(root=self.root_dir, train=True, download=True)
        )
        self.train_dl_b = DataLoader(
            CIFAR10(root=self.root_dir, train=True, download=False)
        )
        self.val_dl_a = DataLoader(
            CIFAR10(root=self.root_dir, train=False, download=False)
        )
        self.val_dl_b = DataLoader(
            CIFAR10(root=self.root_dir, train=False, download=False)
        )

    def eval_fn(self, model: nn.Module, device: str | torch.device) -> float:
        losses: list[float] = []
        loss_fn = nn.CrossEntropyLoss()
        val_dl = self.val_dl_a if model is self.model_a else self.val_dl_b
        iters = self.hparams.percent_eval * len(val_dl) / self.hparams.batch_size

        for i, (inputs, labels) in enumerate(val_dl):
            if i == iters:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def measure_weight_matching(
            self, constructor: Any, weights_a: Any, weights_b: Any, verbose: bool
    ) -> None:
        # Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.set_dataloaders(weights_a, weights_b)

        self.model_a = constructor(weights=weights_a).to(device)
        self.model_b = constructor(weights=weights_b).to(device)

        # They are trained on ImageNet but evaluated on CIFAR10 here
        #   -> recalculate the BatchNorms
        if not self.hparams.ignore_bn:
            recalculate_batch_norms(self.model_a, self.train_dl_a, 0, device, verbose)
            recalculate_batch_norms(self.model_b, self.train_dl_b, 0, device, verbose)

        self.original_model_b = copy.deepcopy(self.model_b)

        results: dict[str, list[float]] = {
            "a_b_original": [], "a_b_rebasin": [], "b_original_b_rebasin": []
        }

        # Rebasin
        if verbose:
            print("\nRebasin")
        input_data, _ = next(iter(self.train_dl_b))
        rebasin = PermutationCoordinateDescent(
            self.model_a,
            self.model_b,
            input_data=input_data,
            device_b=device,
            verbose=verbose
        )
        rebasin.calculate_permutations()
        rebasin.apply_permutations()
        if not self.hparams.ignore_bn:
            recalculate_batch_norms(
                self.model_b,
                self.train_dl_b,
                input_indices=0,
                device=device,
                verbose=verbose
            )

        if verbose:
            print("Interpolate between model_a and model_b (original weights)")

        # Interpolate between original models
        lerp = LerpSimple(
            models=(self.model_a, self.original_model_b),
            devices=[device, device],
            device_interp=device,
            eval_fn=self.eval_fn,
            verbose=verbose
        )
        lerp.interpolate(steps=20)
        results["a_b_original"] = lerp.metrics_interpolated
        loss_a = lerp.metrics_models[0]
        loss_b_original = lerp.metrics_models[1]

        # Interpolate between models with rebasin
        if verbose:
            print("\nInterpolate between model_a and model_b (rebasin weights)")
        lerp = LerpSimple(
            models=(self.model_a, self.model_b),
            devices=[device, device],
            device_interp=device,
            eval_fn=self.eval_fn,
            train_dataloader=self.train_dl_b if not self.hparams.ignore_bn else None,
            verbose=verbose
        )
        lerp.interpolate(steps=20)
        results["a_b_rebasin"] = lerp.metrics_interpolated
        loss_b_rebasin = lerp.metrics_models[1]

        # Interpolate between original and rebasin models
        if verbose:
            print("\nInterpolate between original model_b and rebasin model_b")
        lerp = LerpSimple(
            models=(self.original_model_b, self.model_b),
            devices=[device, device],
            device_interp=device,
            eval_fn=self.eval_fn,
            train_dataloader=self.train_dl_b if not self.hparams.ignore_bn else None,
            verbose=verbose
        )
        lerp.interpolate(steps=20)
        results["b_original_b_rebasin"] = lerp.metrics_interpolated

        # Save results
        # csv saves rows; have to manually transpose
        rows = [
            ["model", *list(results.keys())],
            ("start", loss_a, loss_a, loss_b_original)
        ]

        i = 1
        for l1, (l2, l3) in zip(  # noqa
                results["a_b_original"],
                zip(results["a_b_rebasin"], results["b_original_b_rebasin"])  # noqa
        ):
            rows.append([f"interpolated_{i}", l1, l2, l3])
            i += 1

        rows.append(("end", loss_b_original, loss_b_rebasin, loss_b_rebasin))

        savefile = os.path.join(self.results_dir, f"{constructor.__name__}.csv")
        with open(savefile, "w") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def set_dataloaders(self, weights_a: Any, weights_b: Any) -> None:
        self.train_dl_a = DataLoader(
            CIFAR10(
                self.root_dir,
                download=False,
                train=True,
                transform=weights_a.transforms(),
            ),
            shuffle=True,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )
        self.train_dl_b = DataLoader(
            CIFAR10(
                self.root_dir,
                download=False,
                train=True,
                transform=weights_b.transforms(),
            ),
            shuffle=True,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )
        self.val_dl_a = DataLoader(
            CIFAR10(
                self.root_dir,
                download=False,
                train=False,
                transform=weights_a.transforms()
            ),
            shuffle=False,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )
        self.val_dl_b = DataLoader(
            CIFAR10(
                self.root_dir,
                download=False,
                train=False,
                transform=weights_b.transforms()
            ),
            shuffle=False,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )

    def run(self) -> None:
        """Run the evaluation.

        Attention! All models have BatchNorm layers. This means that the
        interpolation will take long time, because their statistics have
        to be recalculated (meaning that the entire training set has to be
        passed through the model for every interpolation step).
        """
        mandw = [
            minfo for minfo in MODELS_AND_WEIGHTS
            if minfo.constructor.__name__ in self.hparams.models
        ]

        for i, minfo in enumerate(mandw):
            if self.hparams.verbose:
                print(
                    f"\n\n{i+1}/{len(mandw)}: "
                    f"Measuring weight matching for "
                    f"{minfo.constructor.__name__.upper()}"
                )
            self.measure_weight_matching(
                minfo.constructor,
                minfo.weights_a,
                minfo.weights_b,
                self.hparams.verbose
            )


if __name__ == "__main__":
    TorchvisionEval().run()
