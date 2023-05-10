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
        if len(self) < 2:
            self.input_permutation = self.output_permutation = None
            return

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
    def input_permutation(self, permutation: Permutation) -> None:
        shapes = [path.input_shape for path in self if bool(path)]
        if not all(shape == shapes[0] for shape in shapes):
            return

        for path in self:
            path.input_permutation = permutation

    @property
    def output_permutation(self) -> Permutation | None:
        """The permutation of the output of the last module."""
        raise NotImplementedError

    @output_permutation.setter
    def output_permutation(self, permutation: Permutation) -> None:
        raise NotImplementedError

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
        # Conditions for identity:
        #   1. Connect parallel paths to previous layer (automatically makes all
        #      input_permutations the same).
        #      This has to be done from outside of this model.
        #   2. If any of the input_permutations are None, make all of them None.
        #   3. If any of the output_permutations are None, make all of them None.
        #   4. If any of the input_permutations differ in shape of the relevant axes,
        #      make all of them None.
        #   5. If any of the output_permutations differ in shape of the relevant axes,
        #      make all of them None.
        #   6. Make all output_permutations the same.
        #   7. If there is an empty path, make all output_permutations
        #      equal to the input_permutations (and with that, equal to
        #      the output permutation of the previous layer.
        #      This way, the empty path gets the permuted output of the previous layer,
        #      and the parallel paths also output an equivalently permuted version of
        #      their original output; the output-permutation of the previous layer is
        #      compensated via their input_permutations, which is desirable for
        #      maximum connectivity between the permutations,
        #      while their output_permutations, by mirroring that of the previous layer,
        #      make sure that their total effect is equivalent
        #      to that of the empty path: preserving the output_permutation's effect
        #      from the previous layer.)

    def apply_permutations(self) -> None:
        """Apply the permutations in the path to the model."""
        raise NotImplementedError


class ModelGraph:
    """
    The graph of the model.

    Consists of :class:`LinearPath` and :class:`ParallelPaths`.
    """
