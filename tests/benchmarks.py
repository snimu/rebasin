"""
This file contains tests for benchmarking the compute time of the
different algorithms for re-basin, done on different models.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch
from torch import nn
from torchvision.models import resnet18, resnet50, resnet101  # type: ignore[import]
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

    - ResNet18: 0.549 seconds
    - ResNet50: 2.322 seconds
    - ResNet101: 3.086 seconds
    """
    @staticmethod
    def benchmark_pcd(
            model_name: str,
            model_a: nn.Module,
            model_b: nn.Module,
            input_data: Any,
            iters: int = 100
    ) -> None:
        print(f"Benchmarking PermutationCoordinateDescent on {model_name}...")
        elapsed = 0.0

        for _ in tqdm(range(iters)):
            with Timer() as t:
                pcd = PermutationCoordinateDescent(model_a, model_b, input_data)
                pcd.calculate_permutations()
                pcd.apply_permutations()

            elapsed += t.elapsed

        print(f"{model_name}: {elapsed / iters:.3f} seconds")

    @classmethod
    def test_resnet18(cls, iters: int = 100) -> None:
        cls.benchmark_pcd(
            "ResNet18", resnet18(), resnet18(), torch.randn(1, 3, 224, 224), iters
        )

    @classmethod
    def test_resnet50(cls, iters: int = 100) -> None:
        cls.benchmark_pcd(
            "ResNet50", resnet50(), resnet50(), torch.randn(1, 3, 224, 224), iters
        )

    @classmethod
    def test_resnet101(cls, iters: int = 100) -> None:
        cls.benchmark_pcd(
            "ResNet101", resnet101(), resnet101(), torch.randn(1, 3, 224, 224), iters
        )


if __name__ == "__main__":
    bench = BenchmarkPermutationCoordinateDescent()
    bench.test_resnet18()
    bench.test_resnet50()
    bench.test_resnet101()
