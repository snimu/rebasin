## General

Problem: For large models, `Interpolation` & `PermutationCoordinateDescent` 
might need too much memory for one device.

Solution: 

1. Allow for multi-device work.
2. Allow for giving models by savepath, and only ever load two at once.

```python
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn


class Interpolation:
    def __init__(
            self, 
            model_target: nn.Module, 
            models: Sequence[nn.Module | Path | str],
            target_device: torch.device | str | None = None,
            model_devices: Sequence[torch.device | str] | None = None,
    ) -> None:
        ...


class PermutationCoordinateDescent:
    def __init__(
            self, 
            model_a: nn.Module,
            model_b: nn.Module,
            input_data: Any,  # assumed to be on device_b
            device_a: torch.device | str | None = None,
            device_b: torch.device | str | None = None,
    ) -> None:
        ...
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
