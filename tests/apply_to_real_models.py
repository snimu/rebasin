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
from torchvision.datasets import CIFAR10, ImageNet  # type: ignore[import]

from rebasin import PermutationCoordinateDescent
from rebasin.interpolation import LerpSimple
from rebasin.util import recalculate_batch_norms
from tests.fixtures.mandw import MODEL_NAMES, MODELS_AND_WEIGHTS


class TorchvisionEval:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-a", "--all", action="store_true", default=False)
        parser.add_argument("-b", "--batch_size", type=int, default=64)
        parser.add_argument("-d", "--dataset", type=str, default="cifar10")
        parser.add_argument("-e", "--exclude", type=str, nargs='*')
        parser.add_argument("-i", "--ignore_bn", action="store_true", default=False)
        parser.add_argument("-m", "--models", type=str, nargs='*')
        parser.add_argument(
            "-p", "--percent_eval",
            type=float, default=100,
            help="Percent of data to evaluate on. Between 0 and 100."
        )
        parser.add_argument("-s", "--steps", type=int, default=20)
        parser.add_argument("-v", "--verbose", action="store_true", default=True)

        self.hparams = parser.parse_args()

        assert self.hparams.dataset in ("cifar10", "imagenet"), \
            "--dataset must be cifar10 or imagenet"

        assert self.hparams.steps > 0, "--steps must be at least one!"

        if not self.hparams.all:
            assert self.hparams.models, "Must specify --models or --all"

        if self.hparams.models is None:
            self.hparams.models = []
            assert self.hparams.all, "Must specify --models or --all"

        if self.hparams.all:
            self.hparams.models = MODEL_NAMES

        for model_name in self.hparams.models:
            assert model_name in MODEL_NAMES, \
                f"--models: {model_name} not in MODEL_NAMES"

        for model_name in self.hparams.exclude:
            assert model_name in MODEL_NAMES, \
                f"--exclude: {model_name} not in MODEL_NAMES"

        assert self.hparams.batch_size > 0, "--batch_size must be greater than 0"
        assert 0 < self.hparams.percent_eval <= 100, \
            "--percent_eval must be in ]0, 100]"
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
        if (
                not self.hparams.ignore_bn
                and self.hparams.dataset != "imagenet"  # already trained on imagenet
        ):
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
            train_dataloader=self.train_dl_b if not self.hparams.ignore_bn else None,
            verbose=verbose
        )
        lerp.interpolate(steps=self.hparams.steps)
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
        lerp.interpolate(steps=self.hparams.steps)
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
        lerp.interpolate(steps=self.hparams.steps)
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
        if self.hparams.dataset == "cifar10":
            train_ds_a, train_ds_b, val_ds_a, val_ds_b = self.get_cifar10_dataloaders(
                weights_a, weights_b
            )
        elif self.hparams.dataset == "imagenet":
            train_ds_a, train_ds_b, val_ds_a, val_ds_b = self.get_imagenet_dataloaders(
                weights_a, weights_b
            )
        else:
            raise ValueError(f"Unknown dataset {self.hparams.dataset}")

        self.train_dl_a = DataLoader(
            train_ds_a,
            shuffle=True,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )
        self.train_dl_b = DataLoader(
            train_ds_b,
            shuffle=True,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )
        self.val_dl_a = DataLoader(
            val_ds_a,
            shuffle=False,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )
        self.val_dl_b = DataLoader(
            val_ds_b,
            shuffle=False,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )

    def get_cifar10_dataloaders(
            self, weights_a: Any, weights_b: Any
    ) -> tuple[CIFAR10, CIFAR10, CIFAR10, CIFAR10]:
        train_ds_a = CIFAR10(
            root=self.root_dir,
            train=True,
            download=False,
            transform=weights_a.transforms()
        )
        train_ds_b = CIFAR10(
            root=self.root_dir,
            train=True,
            download=False,
            transform=weights_b.transforms()
        )
        val_ds_a = CIFAR10(
            root=self.root_dir,
            train=False,
            download=False,
            transform=weights_a.transforms()
        )
        val_ds_b = CIFAR10(
            root=self.root_dir,
            train=False,
            download=False,
            transform=weights_b.transforms()
        )
        return train_ds_a, train_ds_b, val_ds_a, val_ds_b

    def get_imagenet_dataloaders(
            self, weights_a: Any, weights_b: Any
    ) -> tuple[ImageNet, ImageNet, ImageNet, ImageNet]:
        train_ds_a = ImageNet(
            root=self.root_dir,
            split="train",
            transform=weights_a.transforms()
        )
        train_ds_b = ImageNet(
            root=self.root_dir,
            split="train",
            transform=weights_b.transforms()
        )
        val_ds_a = ImageNet(
            root=self.root_dir,
            split="val",
            transform=weights_a.transforms()
        )
        val_ds_b = ImageNet(
            root=self.root_dir,
            split="val",
            transform=weights_b.transforms()
        )
        return train_ds_a, train_ds_b, val_ds_a, val_ds_b

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
            and minfo.constructor.__name__ not in self.hparams.exclude
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
