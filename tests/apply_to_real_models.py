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
from torchvision.models import (  # type: ignore[import]
    EfficientNet_B1_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    RegNet_X_1_6GF_Weights,
    RegNet_X_3_2GF_Weights,
    RegNet_X_8GF_Weights,
    RegNet_X_16GF_Weights,
    RegNet_X_32GF_Weights,
    RegNet_X_400MF_Weights,
    RegNet_X_800MF_Weights,
    RegNet_Y_3_2GF_Weights,
    RegNet_Y_16GF_Weights,
    RegNet_Y_32GF_Weights,
    RegNet_Y_400MF_Weights,
    RegNet_Y_800MF_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    efficientnet_b1,
    mobilenet_v3_large,
    mobilenetv2,
    regnet_x_1_6gf,
    regnet_x_3_2gf,
    regnet_x_8gf,
    regnet_x_16gf,
    regnet_x_32gf,
    regnet_x_400mf,
    regnet_x_800mf,
    regnet_y_3_2gf,
    regnet_y_16gf,
    regnet_y_32gf,
    regnet_y_400mf,
    regnet_y_800mf,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
)

from rebasin import PermutationCoordinateDescent
from rebasin.interpolation import LerpSimple
from rebasin.util import recalculate_batch_norms

MODELS_AND_WEIGHTS = (  # All the models with V1 and V2 weights
    (efficientnet_b1, EfficientNet_B1_Weights),
    (mobilenetv2, MobileNet_V2_Weights),
    (mobilenet_v3_large, MobileNet_V3_Large_Weights),
    (regnet_x_8gf, RegNet_X_8GF_Weights),
    (regnet_x_16gf, RegNet_X_16GF_Weights),
    (regnet_x_32gf, RegNet_X_32GF_Weights),
    (regnet_x_400mf, RegNet_X_400MF_Weights),
    (regnet_x_800mf, RegNet_X_800MF_Weights),
    (regnet_x_1_6gf, RegNet_X_1_6GF_Weights),
    (regnet_x_3_2gf, RegNet_X_3_2GF_Weights),
    (regnet_y_16gf, RegNet_Y_16GF_Weights),
    (regnet_y_32gf, RegNet_Y_32GF_Weights),
    (regnet_y_3_2gf, RegNet_Y_3_2GF_Weights),
    (regnet_y_400mf, RegNet_Y_400MF_Weights),
    (regnet_y_800mf, RegNet_Y_800MF_Weights),
    (resnext101_32x8d, ResNeXt101_32X8D_Weights),
    (resnext50_32x4d, ResNeXt50_32X4D_Weights),
    (resnet50, ResNet50_Weights),
    (resnet101, ResNet101_Weights),
    (resnet152, ResNet152_Weights),
    (wide_resnet50_2, Wide_ResNet50_2_Weights),
    (wide_resnet101_2, Wide_ResNet101_2_Weights),
)


class ImageNetEval:
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

        model_names = [model.__name__ for model, _ in MODELS_AND_WEIGHTS]
        if self.hparams.all:
            self.hparams.models = model_names

        for model_name in self.hparams.models:
            assert model_name in model_names, f"{model_name} not in model_names"

        assert self.hparams.batch_size > 0, "Batch size must be greater than 0"
        assert 0 < self.hparams.percent_eval <= 100, "Percent eval must be in ]0, 100]"
        self.hparams.percent_eval = self.hparams.percent_eval / 100

        self.root_dir = os.path.join(os.path.dirname(Path(__file__)), "data")
        self.results_dir = os.path.join(os.path.dirname(Path(__file__)), "results")
        self.train_dl = DataLoader(  # Download the data
            CIFAR10(root=self.root_dir, train=True, download=True)
        )
        self.val_dl = DataLoader(
            CIFAR10(root=self.root_dir, train=False, download=True)
        )

    def eval_fn(self, model: nn.Module, device: str | torch.device) -> float:
        losses: list[float] = []
        loss_fn = nn.CrossEntropyLoss()
        assert self.val_dl is not None  # for mypy

        iters = self.hparams.percent_eval * len(self.val_dl) / self.hparams.batch_size

        for i, (inputs, labels) in enumerate(self.val_dl):
            if i == iters:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def measure_weight_matching(
            self, model_type: Any, weights: Any, verbose: bool
    ) -> None:
        # Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Make sure that the transforms are correct (from specific weights)
        self.train_dl = DataLoader(
            CIFAR10(
                self.root_dir,
                download=False,
                train=True,
                transform=weights.IMAGENET1K_V1.transforms(),
            ),
            shuffle=True,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )
        self.val_dl = DataLoader(
            CIFAR10(
                self.root_dir,
                download=False,
                train=False,
                transform=weights.IMAGENET1K_V1.transforms()
            ),
            shuffle=False,
            num_workers=30,
            batch_size=self.hparams.batch_size,
        )

        model_a = model_type(weights=weights.IMAGENET1K_V2).to(device)
        model_b = model_type(weights=weights.IMAGENET1K_V1).to(device)

        # They are trained on ImageNet but evaluated on CIFAR10 here
        #   -> recalculate the BatchNorms
        if not self.hparams.ignore_bn:
            recalculate_batch_norms(model_a, self.train_dl, 0, device, verbose)
            recalculate_batch_norms(model_b, self.train_dl, 0, device, verbose)

        original_model_b = copy.deepcopy(model_b)

        results: dict[str, list[float]] = {
            "a_b_original": [], "a_b_rebasin": [], "b_original_b_rebasin": []
        }

        # Rebasin
        if verbose:
            print("\nRebasin")
        input_data, _ = next(iter(self.train_dl))
        rebasin = PermutationCoordinateDescent(
            model_a,
            model_b,
            input_data=input_data,
            device_b=device,
            verbose=verbose
        )
        rebasin.calculate_permutations()
        rebasin.apply_permutations()
        if not self.hparams.ignore_bn:
            recalculate_batch_norms(
                model_b, self.train_dl, input_indices=0, device=device, verbose=verbose
            )

        if verbose:
            print("Interpolate between model_a and model_b (original weights)")

        # Interpolate between original models
        lerp = LerpSimple(
            models=(model_a, original_model_b),
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
            models=(model_a, model_b),
            devices=[device, device],
            device_interp=device,
            eval_fn=self.eval_fn,
            train_dataloader=self.train_dl if not self.hparams.ignore_bn else None,
            verbose=verbose
        )
        lerp.interpolate(steps=20)
        results["a_b_rebasin"] = lerp.metrics_interpolated
        loss_b_rebasin = lerp.metrics_models[1]

        # Interpolate between original and rebasin models
        if verbose:
            print("\nInterpolate between original model_b and rebasin model_b")
        lerp = LerpSimple(
            models=(original_model_b, model_b),
            devices=[device, device],
            device_interp=device,
            eval_fn=self.eval_fn,
            train_dataloader=self.train_dl if not self.hparams.ignore_bn else None,
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

        savefile = os.path.join(self.results_dir, f"{model_type.__name__}.csv")
        with open(savefile, "w") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def run(self) -> None:
        """Run the evaluation.

        Attention! All models have BatchNorm layers. This means that the
        interpolation will take long time, because their statistics have
        to be recalculated (meaning that the entire training set has to be
        passed through the model for every interpolation step).
        """
        for i, (model_type, weights) in enumerate(MODELS_AND_WEIGHTS):
            if model_type.__name__ not in self.hparams.models:
                continue

            if self.hparams.verbose:
                print(
                    f"\n\n{i+1}/{len(MODELS_AND_WEIGHTS)}: "
                    f"Measuring weight matching for {model_type.__name__.upper()}"
                )
            self.measure_weight_matching(model_type, weights, self.hparams.verbose )


if __name__ == "__main__":
    ImageNetEval().run()
