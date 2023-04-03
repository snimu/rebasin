## `PermutationCoordinateDescent`

Problem: For large models, interpolation might need to much memory for one device.

Solution: allow multi-device interpolation. To do so, 
have the user give `device_a`, `device_b`, and `device_interp`.

Assumption should be that they are all the same device, but this is not required.
Move the data to the correct device for each situation.

Since interpolation works parameter by parameter, it should be possible to do the following:

```python
import torch
model_a, model_b, model_interp, device_a, device_b, device_interp, perc = ...

# Interpolation
for pa, pb, pinterp in zip(model_a.parameters(), model_b.parameters(), model_interp.parameters()):
    pa_ = pa.to(device_interp)
    pb_ = pb.to(device_interp)
    pinterp.data = torch.lerp(pa_, pb_, perc)


# Permutation finding
w_a, w_b = ...
w_a_ = w_a.to(device_a)
w_b_ = w_b.to(device_b)
cost_matrix = w_a_ @ w_b_
```

The same basic principle of course applies wherever two models interact.


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
