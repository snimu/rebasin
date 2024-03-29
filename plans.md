## Initialization

- Give `ModuleParameters` an `internal_permutations` property,
   so that e.g. `MultiheadAttention` can be permuted internally.
- Split path into `LinearPath` and `ResidualPath`. In the latter, 
   the `output_permutation` of the last layer must be equal to the
   one of the last layer of the prior path. This is because 
   the first `input_permutation` makes up for the last `output_permutation` 
   of the prior path, but not in the residual. 

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
