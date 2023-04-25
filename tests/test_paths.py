"""Test the rebasin.initialization._paths module."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rebasin.initialization._paths import ResidualPath
from rebasin.initialization._permutation import ModuleParameters, Perm


class ExampleModuleParameters:
    @property
    def norm1_lp_mp(self) -> ModuleParameters:
        norm1 = nn.BatchNorm2d(3)
        norm1_mp = ModuleParameters(
            norm1.weight, norm1.weight, "norm1", {0: Perm(torch.arange(3))}
        )
        return norm1_mp

    @property
    def conv1_lp_mp(self) -> ModuleParameters:
        conv1 = nn.Conv2d(3, 4, (3, 3))
        conv1_mp = ModuleParameters(
            conv1.weight,
            conv1.weight,
            "conv1",
            {0: Perm(torch.arange(4)), 1: Perm(torch.arange(3))}
        )
        return conv1_mp

    @property
    def norm2_lp_mp(self) -> ModuleParameters:
        norm2 = nn.BatchNorm2d(4)
        norm2_mp = ModuleParameters(
            norm2.weight, norm2.weight, "norm2", {0: Perm(torch.arange(4))}
        )
        return norm2_mp

    @property
    def conv2_lp_mp(self) -> ModuleParameters:
        conv2 = nn.Conv2d(4, 3, (3, 3))
        conv2_mp = ModuleParameters(
            conv2.weight,
            conv2.weight,
            "conv2",
            {0: Perm(torch.arange(3)), 1: Perm(torch.arange(4))}
        )
        return conv2_mp

    @staticmethod
    def get_perm_num_list(
            path: Sequence[ModuleParameters]
    ) -> list[tuple[str, dict[str, int]]]:
        """
        Get a list of tuples of the form (module_name, {in: int, out: int}).
        'in' and 'out' are the names of the axes; the int is the number of the
        permutation associated with that axis.

        :param path: The path to be converted.
        """
        perm_to_num: dict[int, int] = {}
        perms = []
        num = 0
        for mp in path:
            axis_to_permutation = {}
            for ax, perm in mp.axis_to_permutation.items():
                if id(perm) not in perm_to_num.keys():
                    perm_to_num[id(perm)] = num
                    num += 1

                key = "in" if ax == 1 else "out"
                axis_to_permutation[key] = perm_to_num[id(perm)]

            perms.append((mp.name, axis_to_permutation))
        return perms


class TestResidualPath(ExampleModuleParameters):
    """Test the ~ResidualPath class."""

    def test_merge_two_paths(self) -> None:
        """Test the merging of two paths."""

    def test_merge_long_path_only(self) -> None:
        """Test the merging of a long path only."""
        long_path = [
            self.norm1_lp_mp, self.conv1_lp_mp, self.norm2_lp_mp, self.conv2_lp_mp
        ]
        short_path: list[ModuleParameters] = []
        rp = ResidualPath(long_path, short_path)

        target = [
            ('norm1', {'out': 0}),
            ('conv1', {'out': 1, 'in': 0}),
            ('norm2', {'out': 1}),
            ('conv2', {'out': 0, 'in': 1})
        ]
        assert self.get_perm_num_list(rp.long_path) == target

        # Now test for when the long_path has an uneven number of entries.
        long_path = [
            self.norm1_lp_mp, self.conv1_lp_mp, self.conv2_lp_mp
        ]
        short_path = []
        rp = ResidualPath(long_path, short_path)

        target = [
            ('norm1', {'out': 0}),
            ('conv1', {'out': 1, 'in': 0}),
            ('conv2', {'out': 0, 'in': 1})
        ]
        assert self.get_perm_num_list(rp.long_path) == target


