from __future__ import annotations

from collections.abc import Iterator

from rebasin.modules import (  # type: ignore[attr-defined]
    MODULE_TYPES,
    LayerNormModule,
    OneDimModule,
    Permutation,
    PermutationInfo,
)


class LinearPath:
    """A non-branching path through a model."""

    def __init__(self, *modules: MODULE_TYPES) -> None:
        self.modules = list(modules)
        for m in self.modules:
            assert isinstance(m, MODULE_TYPES), \
                "Linear paths must consist of modules from rebasin.modules."

    def __iter__(self) -> Iterator[MODULE_TYPES]:
        return iter(self.modules)

    def __len__(self) -> int:
        return len(self.modules)

    def __getitem__(self, index: int) -> MODULE_TYPES:
        return self.modules[index]

    def __bool__(self) -> bool:
        return bool(self.modules)

    @property
    def input_permutation(self) -> Permutation | None:
        """The permutation of the input to the first module."""
        return self[0].input_permutation if bool(self) else None

    @input_permutation.setter
    def input_permutation(self, permutation: Permutation | None) -> None:
        if not bool(self):
            return
        perm_pt = 0
        self[perm_pt].input_permutation = permutation

        # nn.Modules with 1D-weights have the same input- and output-permutation,
        # meaning that they have to be connected.
        while (
            perm_pt < len(self) - 1
            and isinstance(self[perm_pt], (OneDimModule, LayerNormModule))
        ):
            self[perm_pt + 1].input_permutation = permutation
            perm_pt += 1

    @property
    def input_shape(self) -> int:
        return int(self[0].input_shape) if bool(self) else 0

    @property
    def output_permutation(self) -> Permutation | None:
        """The permutation of the output of the last module."""
        return self[-1].output_permutation if bool(self) else None

    @output_permutation.setter
    def output_permutation(self, permutation: Permutation | None) -> None:
        if not bool(self):
            return

        perm_pt = -1
        self[perm_pt].output_permutation = permutation

        while(
            perm_pt > -len(self)
            and isinstance(self[perm_pt], (OneDimModule, LayerNormModule))
        ):
            self[perm_pt].input_permutation = permutation
            self[perm_pt - 1].output_permutation = permutation
            perm_pt -= 1

    @property
    def output_shape(self) -> int:
        return int(self[-1].output_shape) if bool(self) else 0

    @property
    def permutation_to_info(self) -> list[tuple[Permutation, list[PermutationInfo]]]:
        id_to_perminfo: dict[int, tuple[Permutation, list[PermutationInfo]]] = {}

        for mod in self:
            for permutation, info in mod.permutation_to_info:
                if permutation is None:
                    continue
                if id(permutation) not in id_to_perminfo:
                    id_to_perminfo[id(permutation)] = (permutation, info)
                id_to_perminfo[id(permutation)][1].extend(info)

        return [perm_info for perm_info in id_to_perminfo.values()]

    def enforce_identity(
            self,
            prev_path: LinearPath | ParallelPaths | None = None,
            next_path: LinearPath | ParallelPaths | None = None
    ) -> None:
        """
        Enforce the constraint that the permutations in :class:`LinearPath`
        combined represent the identity function.

        In other words, applying the permutations in :class:`LinearPath`
        changes the layout of the weights and biases of the model, but not
        its output.
        """
        if prev_path is None:
            self.input_permutation = None
        else:
            self.input_permutation = prev_path.output_permutation

        pt0, pt1 = 0, 1
        while pt1 < len(self):
            if self[pt0].output_permutation is None:
                pt0 += 1
                if pt0 == pt1:
                    pt1 += 1
                continue
            elif self[pt1].input_permutation is None:
                pt1 += 1
                continue

            self[pt1].input_permutation = self[pt0].output_permutation
            pt0 += 1
            pt1 += 1

        if next_path is None:
            self.output_permutation = None

    def apply_permutations(self) -> None:
        """Apply the permutations in the path to the model."""
        for mod in self:
            mod.apply_permutations()

    def __repr__(self) -> str:
        modules_strings = [
            f"  {mod.__class__.__name__}("
            + f"\n    module_type: {mod.module_type.__name__}"
            + (
                f"\n    input_shape: {mod.input_shape}"
                if mod.input_permutation is not None
                else "\n    input_shape: None"
            ) + (
                f"\n    output_shape: {mod.output_shape}"
                if mod.output_permutation is not None
                else "\n    output_shape: None"
            ) + f"\n  )\n\n"
            for mod in self
        ]

        if not modules_strings:
            return "\nLinearPath()\n"

        modules_strings = ["LinearPath(", *modules_strings] + [")"]

        # Normalize line width
        max_line_width = (
            max(len(line) for lines in modules_strings for line in lines.splitlines())
        )
        for i, lines in enumerate(modules_strings):
            normalized_lines = [
                line + " " * (max_line_width - len(line))
                for line in lines.splitlines()
            ]
            modules_strings[i] = "\n" + "\n".join(normalized_lines)

        return "".join(modules_strings)


class ParallelPaths:
    """
    Parallel paths.

    Paths through a model that all start and end in the same node
    are parallel paths.They consist of :class:`LinearPath` objects.
    """

    def __init__(self, *paths: LinearPath):
        self.paths = list(paths)
        for p in self.paths:
            assert isinstance(p, LinearPath), \
                "Parallel paths must consist of linear paths."

    def __iter__(self) -> Iterator[LinearPath]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> LinearPath:
        return self.paths[index]

    @property
    def input_permutation(self) -> Permutation | None:
        """
        The permutation of the input to the first module.

        If all input_permutations of the paths (except for empty paths)
        are the same, it returns the common permutation (even if it is :code:`None`),
        else it returns :code:`None`.
        """
        perms = [path.input_permutation for path in self if bool(path)]
        if not all(perm is perms[0] for perm in perms):
            return None
        return perms[0]

    @input_permutation.setter
    def input_permutation(self, permutation: Permutation | None) -> None:
        if permutation is None:
            self._set_all_input_permutations(permutation)
            return

        shapes = [path.input_shape for path in self if bool(path)]
        if (
                permutation.perm_indices.shape[0] == shapes[0]
                and all(shape == shapes[0] for shape in shapes)
        ):
            self._set_all_input_permutations(permutation)

    def _set_all_input_permutations(self, permutation: Permutation | None) -> None:
        for path in self:
            path.input_permutation = permutation

    @property
    def output_permutation(self) -> Permutation | None:
        """The permutation of the output of the last module."""
        perms = [path.output_permutation for path in self if bool(path)]
        if not all(perm is perms[0] for perm in perms):
            return None
        return perms[0]

    @output_permutation.setter
    def output_permutation(self, permutation: Permutation) -> None:
        if permutation is None:
            self._set_all_output_permutations(permutation)
            return

        shapes = [path.output_shape for path in self if bool(path)]
        if (
                permutation.perm_indices.shape[0] == shapes[0]
                and all(shape == shapes[0] for shape in shapes)
        ):
            self._set_all_output_permutations(permutation)

    def _set_all_output_permutations(self, permutation: Permutation | None) -> None:
        for path in self:
            path.output_permutation = permutation

    @property
    def permutation_to_info(self) -> list[tuple[Permutation, list[PermutationInfo]]]:
        id_to_perminfo: dict[int, tuple[Permutation, list[PermutationInfo]]] = {}

        for path in self:
            for permutation, info in path.permutation_to_info:
                if permutation is None:
                    continue
                if id(permutation) not in id_to_perminfo:
                    id_to_perminfo[id(permutation)] = (permutation, info)
                id_to_perminfo[id(permutation)][1].extend(info)

        return [perm_info for perm_info in id_to_perminfo.values()]

    def enforce_identity(
            self,
            prev_path: LinearPath | ParallelPaths | None = None,
            next_path: LinearPath | ParallelPaths | None = None
    ) -> None:
        """
        Enforce the constraint that the permutations in :class:`ParallelPaths`
        represent the identity function in sum.

        In other words, applying the permutations in :class:`ParallelPaths`
        changes the layout of the weights and biases of the model, but not
        its output.
        """
        for path in self:
            path.enforce_identity(prev_path=prev_path, next_path=next_path)

        # So far: input_permutation is None if prev_path is None,
        #   output_permutation is None if next_path is None.
        # Also, input_permutation is the previous output_permutation
        #   if prev_path is not None,
        #   output_permutation is the next input_permutation if next_path is not None.
        # Next, handle the edge-cases:
        #
        # 1. If both prev_path and next_path are None,
        #       then we are done (input_permutation and output_permutation are None).
        # 2. If there is no empty path in self,
        #       then self.output_permutation has to be synchronized
        #       between the paths in self.
        #       The may be short paths where the output_permutation
        #       is the input_permutation, which is None,
        #       in which case all output_permutations have to be set to None
        #       Otherwise, either all output_permutations are None,
        #       in which case setting all to the output_permutation of the first path
        #       works fine, or none of them are None, in which case it is fine
        #       to set all output_permutations to any one of the output permutations,
        #       as long as they are all the same.
        # 3. If there is an empty path in self, and prev_path is None,
        #       then output_permutation should also be set to None.
        # 4. If there is an empty path in self, and next_path is None,
        #       then input_permutation needs to be None,
        #       and so does prev_path.output_permutation.
        # 5. If there is an empty path in self,
        #       and neither prev_path nor next_path are None,
        #       then self.output_permutation has to be prev_path.output_permutation.

        # 1. Done
        if prev_path is None and next_path is None:
            return

        # No empty path in self
        if all(bool(path) for path in self):
            # 2. Synchronize output_permutation
            if any(path.output_permutation is None for path in self):
                self.output_permutation = None
            else:
                self.output_permutation = self[0].output_permutation
            return

        # Empty path in self
        # 3. Set output_permutation to None
        if prev_path is None:
            self.output_permutation = None
            return

        # 4. Set input_permutation and prev_path.output_permutation to None
        if next_path is None:
            self.input_permutation = None
            prev_path.output_permutation = None
            return

        # 5. Synchronize output_permutation
        self.output_permutation = prev_path.output_permutation

    def apply_permutations(self) -> None:
        """Apply the permutations in the path to the model."""
        for path in self:
            path.apply_permutations()

    def __repr__(self) -> str:
        path_strings = [repr(path) for path in self]
        max_len = max(
            len(path_string.splitlines()) for path_string in path_strings)

        for i, path_string in enumerate(path_strings):
            # Pad the path string with spaces
            #   so that the parallel paths don't touch.
            path_strings[i] = "\n".join(
                line + " "*4
                for line in path_string.splitlines()
            )

            # Add "|"-lines to the bottom of the path_string
            #   to make it as long as the longest path.
            # The "|" should be padded by spaces such that it is centered
            #   in a line of length max_width.
            max_width = max(len(line) for line in path_string.splitlines())
            centered_line = (
                    "\n"
                    + " " * (max_width // 2 - 1)
                    + "|"
                    + " " * (max_width // 2 + 3)  # adjust for padding on the right
            )
            path_strings[i] += centered_line * (max_len - len(path_string.splitlines()))

        # Join the path strings line-by-line
        final_path_strings = ["  " for _ in range(max_len)]
        for path_string in path_strings:
            for i, line in enumerate(path_string.splitlines()):
                final_path_strings[i] += line

        return "ParallelPaths(" + "\n".join(final_path_strings) + "\n)"



class ModelGraph:
    """
    The graph of the model.

    Consists of :class:`LinearPath` and :class:`ParallelPaths`.
    """
    def __init__(self, *paths: LinearPath | ParallelPaths) -> None:
        self.paths = list(paths)
        assert all(
            isinstance(path, (LinearPath, ParallelPaths))
            for path in self.paths
        ), "paths must be instances of LinearPath or ParallelPaths"

    def __iter__(self) -> Iterator[LinearPath | ParallelPaths]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> LinearPath | ParallelPaths:
        return self.paths[index]

    @property
    def input_permutation(self) -> Permutation | None:
        return self[0].input_permutation

    @input_permutation.setter
    def input_permutation(self, permutation: Permutation | None) -> None:
        self[0].input_permutation = permutation

    @property
    def output_permutation(self) -> Permutation | None:
        return self[-1].output_permutation

    @output_permutation.setter
    def output_permutation(self, permutation: Permutation | None) -> None:
        self[-1].output_permutation = permutation

    @property
    def permutation_to_info(self) -> list[tuple[Permutation, list[PermutationInfo]]]:
        id_to_perminfo: dict[int, tuple[Permutation, list[PermutationInfo]]] = {}

        for path in self:
            for permutation, info in path.permutation_to_info:
                if permutation is None:
                    continue
                if id(permutation) not in id_to_perminfo:
                    id_to_perminfo[id(permutation)] = (permutation, info)
                id_to_perminfo[id(permutation)][1].extend(info)

        return [perm_info for perm_info in id_to_perminfo.values()]

    def enforce_identity(self) -> None:
        """Enforce the identity constraint on all paths."""
        for i in range(len(self)):
            prev_path = self[i - 1] if i > 0 else None
            next_path = self[i + 1] if i < len(self) - 1 else None
            self[i].enforce_identity(prev_path=prev_path, next_path=next_path)

    def apply_permutations(self) -> None:
        """Apply the permutations in the paths to the model."""
        for path in self:
            path.apply_permutations()

    def __repr__(self) -> str:
        reprs: list[str] = []
        for i, path in enumerate(self):
            width = max(len(line) for line in repr(path).splitlines())
            if i > 0:
                reprs.append(" " * (width // 2 - 1) + "|" + " " * (width // 2 - 1))
                reprs.append(" " * (width // 2 - 1) + "|" + " " * (width // 2 - 1))
                reprs.append(" " * (width // 2 - 1) + "|" + " " * (width // 2 - 1))
            reprs.append("-" * width)
            reprs.append(repr(path))
            reprs.append("-" * width)

        return "ModelGraph(\n" + "\n".join(reprs) + "\n)"
