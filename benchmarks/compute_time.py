"""
This file contains tests for benchmarking the compute time of the
different algorithms for re-basin, done on different models.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch
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

    - ResNet18: 0.551 seconds
    - ResNet50: 2.013 seconds
    - ResNet101: Exceeded maximum recursion depth... # TODO: Fix this
    """
    @staticmethod
    def test_resnet18(iters: int = 100) -> None:
        print("Benchmarking PermutationCoordinateDescent on ResNet18...")
        elapsed = 0.0

        for _ in tqdm(range(iters)):
            model_a = resnet18()
            model_b = resnet18()

            with Timer() as t:
                pcd = PermutationCoordinateDescent(
                    model_a, model_b, torch.randn(1, 3, 224, 224)
                )
                pcd.calculate_permutations()
                pcd.apply_permutations()

            elapsed += t.elapsed

        print(f"ResNet18: {elapsed / iters:.3f} seconds")

    @staticmethod
    def test_resnet50(iters: int = 100) -> None:
        print("Benchmarking PermutationCoordinateDescent on ResNet50...")
        elapsed = 0.0

        for _ in tqdm(range(iters)):
            model_a = resnet50()
            model_b = resnet50()

            with Timer() as t:
                pcd = PermutationCoordinateDescent(
                    model_a, model_b, torch.randn(1, 3, 224, 224)
                )
                pcd.calculate_permutations()
                pcd.apply_permutations()

            elapsed += t.elapsed

        print(f"ResNet50: {elapsed / iters:.3f} seconds")

    @staticmethod
    def test_resnet101(iters: int = 100) -> None:
        print("Benchmarking PermutationCoordinateDescent on ResNet101...")
        elapsed = 0.0

        for _ in tqdm(range(iters)):
            model_a = resnet101()
            model_b = resnet101()

            with Timer() as t:
                pcd = PermutationCoordinateDescent(
                    model_a, model_b, torch.randn(1, 3, 224, 224)
                )
                pcd.calculate_permutations()
                pcd.apply_permutations()

            elapsed += t.elapsed

        print(f"ResNet101: {elapsed / iters:.3f} seconds")


if __name__ == "__main__":
    BenchmarkPermutationCoordinateDescent.test_resnet50()
