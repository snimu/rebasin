"""
This file contains tests for benchmarking the compute time of the
different algorithms for re-basin, done on different models.
"""

from __future__ import annotations

import csv
from time import perf_counter
from typing import Any

import torch
from fixtures.models import MLP, mlp_3b  # type: ignore[import]
from torchvision.models import (  # type: ignore[import]
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from tqdm import tqdm

from rebasin import PermutationCoordinateDescent


class Timer:
    def __enter__(self) -> Timer:
        self.start = perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        del args
        self.stop = perf_counter()
        self.elapsed = self.stop - self.start


class BenchmarkPermutationCoordinateDescent:
    """
    Benchmark the compute time of the PermutationCoordinateDescent algorithm
    on different models.

    The following models are benchmarked (parameter counts measured using torchinfo):

    - ResNet18: 11,689,512 (11.7 million) parameters
    - ResNet34: 21,797,672 (21.8 million) parameters
    - ResNet50: 25,557,032 (25.6 million) parameters
    - ResNet101: 44,549,160 (44.5 million) parameters
    - ResNet152: 60,192,808 (60.2 million) parameters
    - MLP3B: 2,893,400,000 (2.89 billion) parameters

    ---------------
    --- RESULTS ---
    ---------------

    Results on a 2019 MacBook Air (1.6 GHz Intel Core i5, 8 GB 2133 MHz LPDDR3):

    - ResNet18: 1.057 seconds
    - ResNet34: 1.916 seconds
    - ResNet50: 3.526 seconds
    - ResNet101: 5.361 seconds
    - ResNet152: 6.852 seconds

    Results on an A10 GPU (24 GB PCIe, 30vCPU with 200GB RAM):

    - ResNet18: 0.342 seconds
    - ResNet34: 0.565 seconds
    - ResNet50: 1.197 seconds
    - ResNet101: 1.792 seconds
    - ResNet152: 2.252 seconds

    - MLP3B: ?.??? seconds
        Of which:
        - Initialization: ?.??? seconds
        - Calculate permutations: ?.??? seconds
        - Apply permutations: ?.??? seconds
    """
    @staticmethod
    def benchmark_pcd(
            model_name: str,
            model_a_type: Any,
            model_b_type: Any,
            input_data: Any,
            iters: int = 100,
            device_a: str = "cpu",
            device_b: str = "cpu",
            savefile: str | None = None
    ) -> None:
        print(f"Benchmarking PermutationCoordinateDescent on {model_name}...")
        elapsed = 0.0
        all_elapsed: list[float] = []

        for _ in tqdm(range(iters)):
            with Timer() as t:
                # Recreate the models each time to get a representative sample,
                #   and to force permuting the weights.
                # (If the models stay the same, at iteration 2,
                #   model_b will already be permuted,
                #   and no real permutation will occur.)
                pcd = PermutationCoordinateDescent(
                    model_a_type(),
                    model_b_type(),
                    input_data,
                    device_a=device_a,
                    device_b=device_b
                )
                pcd.calculate_permutations()
                pcd.apply_permutations()

            elapsed += t.elapsed
            all_elapsed.append(t.elapsed)

        print(f"{model_name}: {elapsed / iters:.3f} seconds")

        if savefile is not None:
            assert savefile.endswith(".csv")

            with open(savefile, "w") as file:
                writer = csv.writer(file)
                writer.writerow(list(range(iters)))
                writer.writerow(all_elapsed)

    @classmethod
    def test_resnet18(
            cls,
            iters: int = 100,
            device: str = "cpu",
            savefile: str | None = None
    ) -> None:
        cls.benchmark_pcd(
            "ResNet18",
            resnet18,
            resnet18,
            torch.randn(1, 3, 224, 224, device=device),
            iters,
            device,
            device,
            savefile,
        )

    @classmethod
    def test_resnet34(
            cls,
            iters: int = 100,
            device: str = "cpu",
            savefile: str | None = None
    ) -> None:
        cls.benchmark_pcd(
            "ResNet34",
            resnet34,
            resnet34,
            torch.randn(1, 3, 224, 224, device=device),
            iters,
            device,
            device,
            savefile,
        )

    @classmethod
    def test_resnet50(
            cls,
            iters: int = 100,
            device: str = "cpu",
            savefile: str | None = None
    ) -> None:
        cls.benchmark_pcd(
            "ResNet50",
            resnet50,
            resnet50,
            torch.randn(1, 3, 224, 224, device=device),
            iters,
            device,
            device,
            savefile,
        )

    @classmethod
    def test_resnet101(
            cls,
            iters: int = 100,
            device: str = "cpu",
            savefile: str | None = None
    ) -> None:
        cls.benchmark_pcd(
            "ResNet101",
            resnet101,
            resnet101,
            torch.randn(1, 3, 224, 224, device=device),
            iters,
            device,
            device,
            savefile,
        )

    @classmethod
    def test_resnet152(
            cls,
            iters: int = 100,
            device: str = "cpu",
            savefile: str | None = None
    ) -> None:
        cls.benchmark_pcd(
            "ResNet152",
            resnet152,
            resnet152,
            torch.randn(1, 3, 224, 224, device=device),
            iters,
            device,
            device,
            savefile,
        )

    @classmethod
    def test_mlp_3b(cls, savefile: str | None = None) -> None:
        model_a = mlp_3b().to("cpu")
        model_b = mlp_3b().to("cuda")

        with Timer() as t_full:
            pcd = PermutationCoordinateDescent(
                model_a=model_a,
                model_b=model_b,
                input_data_b=torch.randn(1, 3, 224, 224, device="cuda"),
                input_data_a=torch.randn(1, 3, 224, 224, device="cpu"),
                logging_level="info",
                device_a="cpu",
                device_b="cuda"
            )
            print(f"Initialization: {t_full.start - perf_counter():.3f} seconds")

            with Timer() as t_calc:
                pcd.calculate_permutations()
            print(f"Calculate permutations: {t_calc.elapsed:.3f} seconds")

            with Timer() as t_apply:
                pcd.apply_permutations()
            print(f"Apply permutations: {t_apply.elapsed:.3f} seconds")

        print(f"Large MLP: {t_full.elapsed:.3f} seconds")
        if savefile is not None:
            assert savefile.endswith(".csv")

            with open(savefile, "w") as file:
                writer = csv.writer(file)
                writer.writerow(["Full", "Calc", "Apply"])
                writer.writerow([t_full.elapsed, t_calc.elapsed, t_apply.elapsed])


if __name__ == "__main__":
    bench = BenchmarkPermutationCoordinateDescent()
    iters = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench.test_resnet18(iters, device)
    bench.test_resnet34(iters, device)
    bench.test_resnet50(iters, device)
    bench.test_resnet101(iters, device)
    bench.test_resnet152(iters, device)

    if torch.cuda.is_available():
        bench.test_mlp_3b(savefile="results.csv")
