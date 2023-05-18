"""
Apply rebasin to real models.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import]
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageNet  # type: ignore[import]

from rebasin import PermutationCoordinateDescent
from rebasin.interpolation import LerpSimple
from rebasin.utils import recalculate_batch_norms
from tests.fixtures.mandw import MODEL_NAMES, MODELS_AND_WEIGHTS
from tests.fixtures.utils import accuracy


class TorchvisionEval:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-a", "--all",
            action="store_true", default=False,
            help="If this flag is set, all models are evaluated. "
                 "Overrides --models. "
                 "Is overridden by --exclude, meaning that if --all is set "
                 "and --exclude is not empty, then all models except those "
                 "in --exclude are evaluated."
        )
        parser.add_argument(
            "-b", "--batch_size",
            type=int, default=64,
            help="Batch size to use for evaluation."
        )
        parser.add_argument(
            "-d", "--dataset",
            type=str, default="cifar10",
            help="Dataset to evaluate on. Must be cifar10 or imagenet."
        )
        parser.add_argument(
            "--dataset_percentage",
            type=float, default=100.0,
            help="Percentage of dataset to use for recalculating batchnorm. "
                 "Between 0 and 100.",
        )
        parser.add_argument(
            "-e", "--exclude",
            type=str, nargs='*',
            help="Models to exclude from evaluation."
        )
        parser.add_argument(
            "-i", "--ignore_bn",
            action="store_true", default=False,
            help="If this flag is set, BatchNorm statistics are not recalculated."
        )
        parser.add_argument(
            "-m", "--models",
            type=str, nargs='*',
            help="Models to evaluate. If --all is True, then this is ignored.")
        parser.add_argument(
            "-p", "--percent_eval",
            type=float, default=100,
            help="Percent of data to evaluate on. Between 0 and 100."
        )
        parser.add_argument(
            "-s", "--steps",
            type=int, default=20,
            help="Number of interpolation steps per interpolation."
        )
        parser.add_argument(
            "-v", "--verbose",
            action="store_true", default=True,
            help="If this flag is set, progress is printed."
        )

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

        self.hparams.exclude = self.hparams.exclude or []
        for model_name in self.hparams.exclude:
            assert model_name in MODEL_NAMES, \
                f"--exclude: {model_name} not in MODEL_NAMES"

        assert self.hparams.batch_size > 0, "--batch_size must be greater than 0"
        assert 0 < self.hparams.percent_eval <= 100, \
            "--percent_eval must be in ]0, 100]"
        self.hparams.percent_eval = self.hparams.percent_eval / 100

        assert isinstance(self.hparams.dataset_percentage, float), \
            "--dataset_percentage must be a float"
        assert 0 < self.hparams.dataset_percentage <= 100, \
            "--dataset_percentage must be in ]0, 100]"
        self.hparams.dataset_percentage = self.hparams.dataset_percentage / 100.0

        self.root_dir = os.path.join(os.path.dirname(Path(__file__)), "data")
        self.results_dir = os.path.join(os.path.dirname(Path(__file__)), "results")

        if self.hparams.verbose:
            print("Setting dataloaders...")
        self.set_dataloaders()
        if self.hparams.verbose:
            print("Done.")

        self.accuracies1: list[float] = []
        self.accuracies5: list[float] = []
        self.losses: list[float] = []

    def eval_fn(self, model: nn.Module, device: str | torch.device) -> float:
        losses: list[float] = []
        loss_fn = nn.CrossEntropyLoss()
        accuracies1: list[float] = []
        accuracies5: list[float] = []
        iters = self.hparams.percent_eval * len(self.val_dl)

        for i, (inputs, labels) in enumerate(self.val_dl):
            if i == iters:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            accuracies1.append(*accuracy(outputs, labels, topk=(1,)))
            accuracies5.append(*accuracy(outputs, labels, topk=(5,)))

        self.accuracies1.append(sum(accuracies1) / len(accuracies1))
        self.accuracies5.append(sum(accuracies5) / len(accuracies5))
        avg_loss = sum(losses) / len(losses)
        self.losses.append(avg_loss)
        return avg_loss

    def measure_weight_matching(
            self, constructor: Any, weights_a: Any, weights_b: Any, verbose: bool
    ) -> None:
        # Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if verbose:
            print("Setting up models...")
        self.model_a = constructor(weights=weights_a).eval().to(device)
        self.model_b = constructor(weights=weights_b).eval().to(device)

        if (
                not self.hparams.ignore_bn
                and self.hparams.dataset != "imagenet"  # already trained on imagenet
        ):
            if verbose:
                print("Recalculating BatchNorms for original models...")
            self.set_transforms(weights_a)
            recalculate_batch_norms(self.model_a, self.train_dl, 0, device, verbose)
            self.set_transforms(weights_b)
            recalculate_batch_norms(self.model_b, self.train_dl, 0, device, verbose)

        self.original_model_b = copy.deepcopy(self.model_b)

        losses: dict[str, list[float]] = {
            "a_b_original": [], "a_b_rebasin": [], "b_original_b_rebasin": []
        }
        accuracies1: dict[str, list[float]] = {
            "a_b_original": [], "a_b_rebasin": [], "b_original_b_rebasin": []
        }
        accuracies5: dict[str, list[float]] = {
            "a_b_original": [], "a_b_rebasin": [], "b_original_b_rebasin": []
        }
        self.set_transforms(weights_a)
        self.eval_fn(self.model_a, device)
        self.set_transforms(weights_b)
        self.eval_fn(self.model_b, device)
        ma_loss = self.losses[0]
        mb_orig_loss = self.losses[1]
        ma_acc1 = self.accuracies1[0]
        ma_acc5 = self.accuracies5[0]
        mb_orig_acc1 = self.accuracies1[1]
        mb_orig_acc5 = self.accuracies5[1]
        self.losses.clear()
        self.accuracies1.clear()
        self.accuracies5.clear()

        # Rebasin
        if verbose:
            print("\nRebasin")
        self.set_transforms(weights_b)
        input_data, _ = next(iter(self.train_dl))
        rebasin = PermutationCoordinateDescent(
            self.model_a,
            self.model_b,
            input_data_b=input_data.to(device),
            device_b=device,
            logging_level=logging.INFO if verbose else logging.ERROR
        )
        rebasin.rebasin()
        if not self.hparams.ignore_bn:
            recalculate_batch_norms(
                self.model_b,
                self.train_dl,
                input_indices=0,
                device=device,
                verbose=verbose,
                dataset_percentage=self.hparams.dataset_percentage
            )

        self.eval_fn(self.model_b, device)
        mb_rebasin_loss = self.losses[0]
        mb_rebasin_acc1 = self.accuracies1[0]
        mb_rebasin_acc5 = self.accuracies5[0]
        self.losses.clear()
        self.accuracies1.clear()
        self.accuracies5.clear()

        if verbose:
            print("Interpolate between model_a and model_b (original weights)")

        # Interpolate between original models
        lerp = LerpSimple(
            models=(self.model_a, self.original_model_b),
            devices=[device, device],
            device_interp=device,
            eval_fn=self.eval_fn,
            train_dataloader=self.train_dl if not self.hparams.ignore_bn else None,
            logging_level=logging.INFO if verbose else logging.ERROR,
            dataset_percentage=self.hparams.dataset_percentage,
        )
        lerp.interpolate(steps=self.hparams.steps)
        losses["a_b_original"] = [ma_loss, *self.losses] + [mb_orig_loss]
        accuracies1["a_b_original"] = [ma_acc1, *self.accuracies1] + [mb_orig_acc1]
        accuracies5["a_b_original"] = [ma_acc5, *self.accuracies5] + [mb_orig_acc5]
        self.losses.clear()
        self.accuracies1.clear()
        self.accuracies5.clear()

        # Interpolate between models with rebasin
        if verbose:
            print("\nInterpolate between model_a and model_b (rebasin weights)")
        lerp = LerpSimple(
            models=(self.model_a, self.model_b),
            devices=[device, device],
            device_interp=device,
            eval_fn=self.eval_fn,
            train_dataloader=self.train_dl if not self.hparams.ignore_bn else None,
            logging_level=logging.INFO if verbose else logging.ERROR,
            dataset_percentage=self.hparams.dataset_percentage,
        )
        lerp.interpolate(steps=self.hparams.steps)
        losses["a_b_rebasin"] = [ma_loss, *self.losses] + [mb_rebasin_loss]
        accuracies1["a_b_rebasin"] = [ma_acc1, *self.accuracies1] + [mb_rebasin_acc1]
        accuracies5["a_b_rebasin"] = [ma_acc5, *self.accuracies5] + [mb_rebasin_acc5]
        self.losses.clear()
        self.accuracies1.clear()
        self.accuracies5.clear()

        # Interpolate between original and rebasin models
        if verbose:
            print("\nInterpolate between original model_b and rebasin model_b")
        lerp = LerpSimple(
            models=(self.original_model_b, self.model_b),
            devices=[device, device],
            device_interp=device,
            eval_fn=self.eval_fn,
            train_dataloader=self.train_dl if not self.hparams.ignore_bn else None,
            logging_level=logging.INFO if verbose else logging.ERROR,
            dataset_percentage=self.hparams.dataset_percentage,
        )
        lerp.interpolate(steps=self.hparams.steps)
        losses["b_original_b_rebasin"] = \
            [mb_orig_loss, *self.losses] + [mb_rebasin_loss]
        accuracies1["b_original_b_rebasin"] = \
            [mb_orig_acc1, *self.accuracies1] + [mb_rebasin_acc1]
        accuracies5["b_original_b_rebasin"] = \
            [mb_orig_acc5, *self.accuracies5] + [mb_rebasin_acc5]
        self.losses.clear()
        self.accuracies1.clear()
        self.accuracies5.clear()

        # Save results
        if verbose:
            print("\nSaving results...")
        savefile = os.path.join(self.results_dir, f"{constructor.__name__}.csv")
        df_losses = pd.DataFrame(losses)
        df_accuracies1 = pd.DataFrame(accuracies1)
        df_accuracies5 = pd.DataFrame(accuracies5)
        df_losses.to_csv(savefile.replace(".csv", "_losses.csv"))
        df_accuracies1.to_csv(savefile.replace(".csv", "_accuracies1.csv"))
        df_accuracies5.to_csv(savefile.replace(".csv", "_accuracies5.csv"))

        if verbose:
            print("Done")

    def set_dataloaders(self) -> None:
        if self.hparams.dataset == "cifar10":
            train_ds, val_ds = self.get_cifar10_datasets()
        elif self.hparams.dataset == "imagenet":
            train_ds, val_ds = self.get_imagenet_datasets()
        else:
            raise ValueError(f"Unknown dataset {self.hparams.dataset}")

        self.train_dl = DataLoader(
            train_ds,
            shuffle=True,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )
        self.val_dl = DataLoader(
            val_ds,
            shuffle=False,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )

    def get_cifar10_datasets(self) -> tuple[CIFAR10, CIFAR10]:
        train_ds = CIFAR10(
            root=self.root_dir,
            train=True,
            download=True,
        )
        val_ds = CIFAR10(
            root=self.root_dir,
            train=False,
            download=False,
        )
        return train_ds, val_ds

    def get_imagenet_datasets(self) -> tuple[ImageNet, ImageNet]:
        train_ds = ImageNet(
            root=self.root_dir,
            split="train",
        )
        val_ds = ImageNet(
            root=self.root_dir,
            split="val",
        )
        return train_ds, val_ds

    def set_transforms(self, weights: Any) -> None:
        self.train_dl.dataset.transform = (  # type: ignore[attr-defined]
            weights.transforms()
        )
        self.val_dl.dataset.transform = (  # type: ignore[attr-defined]
            weights.transforms()
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
