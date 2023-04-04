"""
This file contains tests for benchmarking the compute time of the
different algorithms for re-basin, done on different models.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch
from fixtures.models import MLP  # type: ignore[import]
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

    Results on a 2019 MacBook Air (1.6 GHz Intel Core i5, 8 GB 2133 MHz LPDDR3):

    - ResNet18: 1.057 seconds
    - ResNet34: 1.916 seconds
    - ResNet50: 3.526 seconds
    - ResNet101: 5.361 seconds
    - ResNet152: 6.852 seconds

    Results on an A10 GPU (24 GB PCIe, 30vCPU with 200GB RAM):

    - Large MLP: 0.000 seconds
        Of which:
        - Initialization: 0.000 seconds
        - Calculate permutations: 0.000 seconds
        - Apply permutations: 0.000 seconds
    """
    @staticmethod
    def benchmark_pcd(
            model_name: str,
            model_a_type: Any,
            model_b_type: Any,
            input_data: Any,
            iters: int = 100
    ) -> None:
        print(f"Benchmarking PermutationCoordinateDescent on {model_name}...")
        elapsed = 0.0

        for _ in tqdm(range(iters)):
            with Timer() as t:
                # Recreate the models each time to get a representative sample,
                #   and to force permuting the weights.
                # (If the models stay the same, at iteration 2,
                #   model_b will already be permuted,
                #   and no real permutation will occur.)
                pcd = PermutationCoordinateDescent(
                    model_a_type(), model_b_type(), input_data
                )
                pcd.calculate_permutations()
                pcd.apply_permutations()

            elapsed += t.elapsed

        print(f"{model_name}: {elapsed / iters:.3f} seconds")

    @classmethod
    def test_resnet18(cls, iters: int = 100) -> None:
        cls.benchmark_pcd(
            "ResNet18", resnet18, resnet18, torch.randn(1, 3, 224, 224), iters
        )

    @classmethod
    def test_resnet34(cls, iters: int = 100) -> None:
        cls.benchmark_pcd(
            "ResNet34", resnet34, resnet34, torch.randn(1, 3, 224, 224), iters
        )

    @classmethod
    def test_resnet50(cls, iters: int = 100) -> None:
        cls.benchmark_pcd(
            "ResNet50", resnet50, resnet50, torch.randn(1, 3, 224, 224), iters
        )

    @classmethod
    def test_resnet101(cls, iters: int = 100) -> None:
        cls.benchmark_pcd(
            "ResNet101", resnet101, resnet101, torch.randn(1, 3, 224, 224), iters
        )

    @classmethod
    def test_resnet152(cls, iters: int = 100) -> None:
        cls.benchmark_pcd(
            "ResNet152", resnet152, resnet152, torch.randn(1, 3, 224, 224), iters
        )

    @classmethod
    def test_large_mlp(cls) -> None:
        model_a = MLP(1000, num_layers=5_000)
        model_b = MLP(1000, num_layers=5_000).to("cuda")

        with Timer() as t_full:
            pcd = PermutationCoordinateDescent(
                model_a, model_b, torch.randn(1000).to("cuda")
            )
            print(f"Initialization: {t_full.start - perf_counter():.3f} seconds")

            with Timer() as t_calc:
                pcd.calculate_permutations()
            print(f"Calculate permutations: {t_calc.elapsed:.3f} seconds")

            with Timer() as t_apply:
                pcd.apply_permutations()
            print(f"Apply permutations: {t_apply.elapsed:.3f} seconds")

        print(f"Large MLP: {t_full.elapsed:.3f} seconds")


if __name__ == "__main__":
    bench = BenchmarkPermutationCoordinateDescent()
    bench.test_resnet18()
    bench.test_resnet34()
    bench.test_resnet50()
    bench.test_resnet101()
    bench.test_resnet152()
