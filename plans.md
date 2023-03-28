## `PermutationCoordinateDescent`

1. Permute the columns of `model_b` **DONE**
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
            self, method: str = "lerp", steps: int = 10, metric: str = "accuracy"
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
          target_model: torch.nn.Module,
          models_to_permute: list[torch.nn.Module],
          train_dataloader: DataLoader,
          val_dataloader: DataLoader
  ) -> None:
    self.target_model = target_model
    self.models_to_permute = models_to_permute
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader

  def rebasin(self, method: str = "weight") -> None:
    assert method in ("weight", "activation", "straight through")
    ...

  def interpolate(
          self,
          savedir: Path,
          save_all: bool = False,
          method: str = "lerp",
          steps: int = 10,
          metric: str = "accuracy"
  ) -> None:
    ...
```
