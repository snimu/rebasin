# rebasin

![PyPI Version](https://img.shields.io/pypi/v/rebasin)
![Wheel](https://img.shields.io/pypi/wheel/rebasin)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-370/)
![License](https://img.shields.io/github/license/snimu/rebasin)

An implementation of methods described in 
["Git Re-basin"-paper by Ainsworth et al.](https://arxiv.org/abs/2209.04836)

Can be applied to **arbitrary models**, without modification.

---

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Installation
```bash
pip install rebasin
```

## Usage

Currently, only weight-matching is implemented as a method for rebasing, 
and only a simplified form of linear interpolation is implemented.

```python
import torch
from torch import nn
from rebasin import PermutationCoordinateDescent
from rebasin import interpolation

model_a, model_b, train_dl, val_dl, loss_fn = ...
device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_fn(model: nn.Module, model_device: str | torch.device | None = None) -> float:
    loss = 0.0
    for inputs, logits in val_dl:
        if model_device is not None:
            inputs = inputs.to(model_device)
            logits = logits.to(model_device)
        outputs = model(inputs)
        loss = loss_fn(outputs, logits)
    return loss / len(val_dl)


input_data = next(iter(train_dl))[0]

# Rebasin
pcd = PermutationCoordinateDescent(model_a, model_b, input_data)
pcd.rebasin()

# Interpolate
lerp = interpolation.LerpSimple(
    models=[model_a, model_b],
    devices=[device, device],
    eval_fn=eval_fn,  # Can be any metric as long as the function takes a model and a device
    eval_mode="min",  # "min" or "max"
    train_dataloader=train_dl,  # Used to recalculate BatchNorm statistics; optional
)
lerp.interpolate(steps=10)

# Access model with lowest validation loss:
lerp.best_model
```

## Terminology

In this document, I will use the following terminology:

- **To rebasin**: To apply one of the methods described in the paper to a model,
    permuting the rows and columns of its weights (and biases)
- `model_a`: The model that stays unchanged
- `model_b`: The model that is changed by rebasin it towards `model_a`
    - `model_b (original)` for the unchanged, original `model_b`
    - `model_b (rebasin)` for the changed, rebasined `model_b`
- **Path**: A linear sequence of modules in a model

## Limitations

### Only some methods are implemented

For rebasin, only weight-matching is implemented via `rebasin.PermutationCoordinateDescent`.

For interpolation, only a simplified method of linear interpolation is implemented 
via `rebasin.interpolation.LerpSimple`.

### Limitations of the `PermutationCoordinateDescent`-class

The `PermutationCoordinateDescent`-class only permutes some Modules. 
Most modules should work, but others may behave unexpectedly. In this case, 
you need to add the module to [rebasin/modules.py](rebasin/modules.py);
make sure it is included in the `initialize_module`-function 
(preferably by putting it into the `SPECIAL_MODULES`-dict).

Additionally, the `PermutationCoordinateDescent`-class only works with
`nn.Module`s, not functions. There is a requirement to have the permuted model
produce the same output as the un-permuted `Module`, which is a pretty 
tight constraint. In some models, it isn't a problem at all, but especially in 
models with lots of short residual blocks, it may (but doesn't have to) be a problem.
Where it is a problem, few to no parameters get permuted, which defeats the purpose of rebasin.

For example, @tysam-code's [hlb-gpt](https://github.com/tysam-code/hlb-gpt), a small but fast
language model implementation, isn't permuted at all. On the other hand,
`torchvision.models.vit_b_16`, which is similarly a transformer-based model,
works perfectly fine (and I will probably produce some results for it at some point).

## Results

For the full results, see [rebasin-results](https://github.com/snimu/rebasin-results)
(I don't want to upload a bunch of images to this repo, so the results are in their own repo).

Here is a little taste:

<p align="center">
    <img
        src="https://github.com/snimu/rebasin-results/blob/main/hlb-CIFAR10/3x3-plot.png"
        alt="hlb-CIFAR10: losses and accuracies of the model"
        width="600"
    />
</p>

It seems to work!

## Acknowledgements

**Git Re-Basin:**

```
Ainsworth, Samuel K., Jonathan Hayase, and Siddhartha Srinivasa. 
"Git re-basin: Merging models modulo permutation symmetries." 
arXiv preprint arXiv:2209.04836 (2022).
```

Link: https://arxiv.org/abs/2209.04836 (accessed on April 9th, 2023)


**ImageNet:**

I've used the ImageNet Data from the 2012 ILSVRC competition to evaluate 
the algorithms from rebasin on the `torchvision.models`.

```
Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, 
Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, 
Alexander C. Berg and Li Fei-Fei. (* = equal contribution) 
ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014
```

[Paper (link)](https://arxiv.org/abs/1409.0575) (Accessed on April 12th, 2023)


**Torchvision models**

For testing, I've used the torchvision models (v.015), of course (or I will): 

https://pytorch.org/vision/0.15/models.html

**HLB-CIFAR10**
For testing, I forked [hlb-CIFAR10](https://github.com/tysam-code/hlb-CIFAR10) 
by [@tysam-code](https://github.com/tysam-code):

    authors:
    - family-names: "Balsam"
      given-names: "Tysam&"
    title: "hlb-CIFAR10"
    version: 0.4.0
    date-released: 2023-02-12
    url: "https://github.com/tysam-code/hlb-CIFAR10"

**HLB-GPT**
For testing, I also used [hlb-gpt](https://github.com/tysam-code/hlb-gpt) by @tysam-code: 

    authors:
      - family-names: "Balsam"
        given-names: "Tysam&"
    title: "hlb-gpt"
    version: 0.0.0
    date-released: 2023-03-05
    url: "https://github.com/tysam-code/hlb-gpt"


**Other**

My code took inspiration from the following sources:

- https://github.com/themrzmaster/git-re-basin-pytorch

I used the amazing library `torchview` to visualize the models:

- https://github.com/mert-kurttutan/torchview

