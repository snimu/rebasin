## General

Initialization of Permutations.

- Use paths
- In linear paths: just connect the previous with the current layer
- Work with `rebasin.Parameter`s, not `Permutation`s!
  - This way, I can easily permute the parameters, 
     and add an `except_axis` argument for `PermutationCoordinateDescent`
  - Different `rebasin.Parameter`s would share one `Permutation` &rarr; 
     I can achieve the `Permutation` sharing, but can handle the parameters 
     more elegantly
- In Residual Paths:
  - Find a mirror-point in the center
  - Mirror the permutations along it
  - This way, the permutations are always symmetric and the residual path is 
     the identity permutation when seen from the rest of the model
  - Example:
     ```markdown
        (P2, P1)
        (P3, P2)
        (P4, P3)
    
        (P4, P4)
    
        (P3, P4)
        (P2, P3)
        (P1, P2)
    ```
    Or, alternatively:
    ```markdown
        (P1, P2)
        (P2, P3)
        (P3, P2)
        (P2, P1)
    ```
    - Then, connect the short path and the long path, if there is something in the short path.
    - Finnally, connect the first and last `rebasin.Parameter` with a 
       `rebasin.LinearPath`


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
