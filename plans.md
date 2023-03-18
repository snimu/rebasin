## `PermutationCoordinateDescent`

- Use column-indices instead of permutation-tensors
  - Makes checking for composability easier
  - Takes less memory
  - Is more readable
  - **TODO**: Figure out how to use this to calculate the permutations
- Find parents / children:
  1. Find all parents / children
  2. Filter out parents / children that are not composable
  3. Save the result in a `dict[id, list[id]]`
- In the permutation loop: use one function 
  to extract the weights and permutation columns
  from the parents / children

## `PermuteColumns`

1. Permute the columns of `model_b`
2. Recompute the `LayerNorm`s by calling `model_b.forward()`
   on every batch in the training dataset (?)

## `Interpolation`

- Must be capable of interpolating between two or more models
- Recompute the `LayerNorm`s by calling `model.forward()`
   on every batch in the training dataset for every interpolated model (?)

```python
from pathlib import Path

import torch
from torch.utils.data import DataLoader


class Interpolation:
    def __init__(
            self, 
            models: list[torch.nn.Module], 
            train_dataloader: DataLoader, 
            val_dataloader: DataLoader,
            savedir: Path,
            save_all: bool = False,
    ) -> None:
        self.models = models
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.savedir = savedir
        self.save_all = save_all  # If True, save all models, else, only the best
        self.results = torch.zeros(len(models))

    def interpolate(
            self, mode: str = "lerp", steps: int = 10, metric: str = "accuracy"
    ) -> None:
        ...
```

## `Rebasin`

The main class for the rebasin algorithm.

```python
from pathlib import Path

import torch
from torch.utils.data import DataLoader


class Rebasin:
    def __init__(
            self, 
            model_target: torch.nn.Module, 
            models: list[torch.nn.Module],
            train_dataloader: DataLoader, 
            val_dataloader: DataLoader
    ) -> None:
        self.model_target = model_target
        self.models = models
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def rebasin(self, method: str = "weight") -> None:
        assert method in ("weight", "activation", "straight through")
        ...

    def interpolate(
            self,
            savedir: Path,
            save_all: bool = False,
            mode: str = "lerp", 
            steps: int = 10, 
            metric: str = "accuracy"
    ) -> None:
        ...
```
