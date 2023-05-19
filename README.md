# rebasin

![PyPI Version](https://img.shields.io/pypi/v/rebasin)
![Wheel](https://img.shields.io/pypi/wheel/rebasin)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-370/)
![License](https://img.shields.io/github/license/snimu/rebasin)

An implementation of methods described in 
["Git Re-basin"-paper by Ainsworth et al.](https://arxiv.org/abs/2209.04836)

Can be applied to **arbitrary models**, without modification. 

(Well, *almost* arbitrary models, see [Limitations](#limitations)).

---

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Installation

Requirements should be automatically installed, but one of them is graphviz, 
which you might have to install per apt / brew / ... on your device.

The following install instructions are taken directly from 
[torchview's installation instructions](https://github.com/mert-kurttutan/torchview#installation).

Debian-based Linux distro (e.g. Ubuntu):

```Bash
apt-get install graphviz
```

Windows:

```Bash
choco install graphviz
```

macOS

```Bash
brew install graphviz
```

see more details [here](https://graphviz.readthedocs.io/en/stable/manual.html).


Then, install rebasin via pip:

```bash
pip install rebasin
```

## Usage

Currently, only weight-matching is implemented as a method for rebasing, 
and only a simplified form of linear interpolation is implemented.

The following is a minimal example. For now, the documentation lives in the docstrings,
though I intend to create a proper one. 
`PermutationCoordinateDescent` and `interpolation.LerpSimple`
are the main classes, beside `MergeMany` (see below).

```python
from rebasin import PermutationCoordinateDescent
from rebasin import interpolation

model_a, model_b, train_dl= ...
input_data = next(iter(train_dl))[0]

# Rebasin
pcd = PermutationCoordinateDescent(model_a, model_b, input_data)  # weight-matching
pcd.rebasin()  # Rebasin model_b towards model_a. Automatically updates model_b

# Interpolate
lerp = interpolation.LerpSimple(
    models=[model_a, model_b],
    devices=["cuda:0", "cuda:1"],  # Optional, defaults to cpu
    device_interp="cuda:2",  # Optional, defaults to cpu
    savedir="/path/to/save/interpolation"  # Optional, save all interpolated models
)
lerp.interpolate(steps=99)  # Interpolate 99 models between model_a and model_b
```

The `MergeMany`-algorithm is also implemented 
(though there will be interface-changes regarding the devices in the future):

```python
from rebasin import MergeMany
from torch import nn

class ExampleModel(nn.Module):
    ...

model_a, model_b, model_c = ExampleModel(), ExampleModel(), ExampleModel()
train_dl = ...

# Merge
merge = MergeMany(
    models=[model_a, model_b, model_c],
    working_model=ExampleModel(),
    input_data=next(iter(train_dl))[0],
)
merged_model = merge.run()
# The merged model is also accessible through merge.working_model,
#   but only after merge.run() has been called.
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
language model implementation, isn't permuted at all. 
Vision transformers like `torchvision.models.vit_b_16` have only very few permutations
applied to them. In general, **transformer models don't work well**, because they 
reshape the input-tensor, and directly follow that up with residual blocks. 
This means that almost nothing of the model can be permuted 
(a single Linear layer between the reshaping and the first residual block would fix that,
but this isn't usually done...).

On the other hand, **CNNs usually work very well**.

If you are unsure, you can always print the model-graph! To do so, write:

```python
from rebasin import PermutationCoordinateDescent


pcd = PermutationCoordinateDescent(...)
print(pcd.pinit.model_graph)  # pinit stands for "PermutationInitialization"
```

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

While `PermutationCoordinateDescent` doesn't fully eliminate the loss-barrier, 
it does reduce it significantly, and, surprisingly, even moreso for the accuracy-barrier.

It seems likely to me that rebasin several models into the same loss-basin will 
produce better results, if it works. This is because for that to work, the 
common loss-basin would have to be a very wide one, which likely leads to better
generalization. This is even one of the claims in the paper. 
I might (*might*) implement the `MergeMany`-algorithm soon.

It is important to point out that BatchNorm is very problematic in a model; it is necessary to recalculate the 
running_stats for a significant number of training batches, which is very compute intensive.
This wasn't a problem for models trained on CIFAR10, but I'm currently struggeling with
the `torchvision.models` trained on ImageNet. I hope to get away with 200 batches of size 64
(for a total of 12,800 images) for recalculating the `BatchNorm`s.
Results will follow (at some point).

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

