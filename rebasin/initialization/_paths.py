"""
The path through the model as parametrized by ~ModuleParameters.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from rebasin.initialization._permutation import ModuleParameters


class ResidualPath:
    """
    A path through the model as parametrized by ~ModuleParameters.
    """
    def __init__(
            self,
            long_path: Sequence[ModuleParameters],
            short_path: Sequence[ModuleParameters]
    ) -> None:
        if not long_path:
            raise ValueError("The long path must be non-empty.")

        self.long_path = long_path
        self.short_path = short_path

        if self.short_path:
            self._merge_two_paths()
        else:
            self._merge_long_path_only()

    def _merge_two_paths(self) -> None:
        """
        If there are permutable layers in the short path, then the long path
        must achieve the same total permutation as the short path.

        This is important for the output of both to sum up correctly.
        """
        raise NotImplementedError

    def _merge_long_path_only(self) -> None:
        """
        The short path simply copies the input to the output of the residual layer.
        In order for the model to sum the residual and the output of the long path
        correctly, after permuting the long path, the total of the permutations of the
        long path must yield the identity permutation.

        For this to happen, we must mirror the permutations along the center
        of the long path.
        """
        # First, merge the path linearly.
        # This is so that the correct permutation is
        self._merge_path_linearly(self.long_path)

        # Then, mirror the path.
        left_pt = 0
        right_pt = -1

        # The output-shape must be equal to the input-shape!
        if (
                len(self.long_path[left_pt].input_permutation)
                != len(self.long_path[right_pt].output_permutation)
        ):
            raise ValueError(
                f"The long-path must be symmetric in shape. "
                f"Specifically, the input-permutation of the left-most module "
                f"must be equal in shape to the output-permutation "
                f"of the right-most module."
                f"It is not: "
                f"{len(self.long_path[left_pt].input_permutation)} != "
                f"{len(self.long_path[right_pt].output_permutation)}"
            )

        while left_pt < len(self.long_path) + right_pt:
            # Merge the output of the right module with the input of the left module.
            self.long_path[right_pt].output_permutation = \
                self.long_path[left_pt].input_permutation

            # Merge the input of the right module with the output of the left module.
            # If the left module's output-shape is not equal to the right module's
            # input-shape, it must be equal to its own input shape. In that case,
            # we can simply merge the input of the right module with the output of
            # the next module on the left.
            while (
                    len(self.long_path[left_pt].output_permutation)
                    != len(self.long_path[right_pt].input_permutation)
            ):
                left_pt += 1

            self.long_path[right_pt].input_permutation = \
                self.long_path[left_pt].output_permutation

            # Proceed to the next modules.
            left_pt += 1
            right_pt -= 1

        # Now, merge normally
        # self._merge_path_linearly(self.long_path)

    @staticmethod
    def _merge_path_linearly(path: Sequence[ModuleParameters]) -> None:
        """
        Merge the path linearly, i.e. merge the output of each module with the
        input of the next module, if they fit.
        """
        for mod0, mod1 in itertools.pairwise(path):
            if len(mod0.output_permutation) != len(mod1.input_permutation):
                continue
            mod1.input_permutation = mod0.output_permutation

