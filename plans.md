## `PermutationCoordinateDescent`

Problem: only permutes weights called "weight" and biases called "bias",
which excludes for example `MultiheadAttention` layers. 
These have weights called "in_proj_weight", "out_proj.weight", etc.

Solution:

```python
# DON'T:
if name == "weight" or name == "bias":
  ...

# DO:
for name, param in model.named_parameters():
  if "weight" in name or "bias" in name:
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
