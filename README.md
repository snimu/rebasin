# rebasin

![PyPI Version](https://img.shields.io/pypi/v/rebasin)
![Wheel](https://img.shields.io/pypi/wheel/rebasin)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
![License](https://img.shields.io/github/license/snimu/rebasin)

An implementation of methods described in 
["Git Re-basin"-paper by Ainsworth et al.](https://arxiv.org/abs/2209.04836)

Can be applied to **arbitrary models**, without modification.

---

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [Results: Weight-matching (PermutationCoordinateDescent)](#results-weight-matching-permutationcoordinatedescent)
  - [HLB-GPT](#hlb-gpt)
  - [torchvision.models](#torchvisionmodels)
    - [Caveats](#caveats)
    - [General takeaways](#general-takeaways)
    - [vit_b_16](#vitb16)
    - [efficientnet_b1](#efficientnetb1)
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
pcd.calculate_permutations()
pcd.apply_permutations()

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

## Limitations

### Only some methods are implemented

Currently, only weight-matching is implemented as a method for rebasing, 
    and only a simplified form of linear interpolation is implemented.



### Limitations of the `PermutationCoordinateDescent`-class

The `PermutationCoordinateDescent`-class only permutes some Modules:

For one thing, it only permutes the weights of modules with a `weight`-attribute.
This means, for example, that `nn.MultiheadAttention` is currently not supported.
There are plans in place to remedy this, but it will take some time.

There is a second limitation, caused by the requirement to have the permuted model
behave the same as the original model.

It splits a network into linear paths. This means, for example, that a residual path
splits the network for the purpose of permutation, into four paths:

 1. The Path up to the residual path.
 2. The main path in the residual path.
 3. The shortcut path.
 4. The path after the residual path.

For each path, the input-permutation of the first module and the output permutation of
the last module in that path are the identity &mdash; they are not permuted.

This is because **each path needs to permute the weights in it in such a way that the
total permutation of that path is the identity**. In other words, the permuted model 
should not change its behavior due to the permutation.

This property limits the number of modules that are permuted. 

Consider the following example:

<p align="center">
  <img 
    src="images/vit_b_16_residual_mlp.png" 
    alt="A residual path including an MLP in ViT_B_16 by torchvision.models" 
    width="500"
  />
</p>

It is a view from the graph of the `vit_b_16`-model from `torchvision.models`
(see [here](images/vit_b_16.pdf) for the full model). 

In it, the only Modules with weights are the two `Linear`-layers. 
This means that the only things getting permuted are the output-axis 
of the weight of the first `Linear`-layer and its bias, and the input-axis of the weight of the second
layer.

In other words, if we name these two `Linear`-layers `Linear1` and `Linear2`,
then `Linear1.weight` at axis 0, `Linear2.weight` at axis 1, and 
`Linear1.bias` are permuted.

Only permuting so few parts of the model might lead to a poor rebasing, because `model_b` 
may be moved only slightly towards `model_a`. 

As a hint to how much this might be the case,
I applied random permutations to `torchvision.models.vit_b_16` with the weights 
`torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1`. The above constraints were in place.
I then calculated the model change (as defined [here](tests/fixtures/util.py))
between the original `model_b` and its rebasined version
It is circa **83.8%**. The output between the original model and the rebasined model
only changes by **4.3e-7** (**4.3e-5%**, or **0.000043%**), as measured by 
`(y_orig - y_new).abs().sum() / y_orig.abs().sum()`. 

The output change is very low, as expected.
However, while the model change is fairly high, it might be interesting to 
see if it could be brought higher. 

To remedy this second issue, I plan to give `PermutationCoordinateDescent` the option 
`enforce_identity: bool = True`. If this is set to `False`, then the permutations
will not be constrained to be the identity at the start and end of each path.

It will be interesting to see if this reduces a model's performance, and if so, by how much.

## Results: Weight-matching (PermutationCoordinateDescent)

Below, I present some preliminary results. In them, I used `torchvision`-models with two different 
pre-trained weights. I interpolated between the two sets of weights, 
"rebasined" one, and interpolated again, saving all the losses.

---

### HLB-GPT

I forked [tysam-code/hlb-gpt](https://github.com/tysam-code/hlb-gpt) 
(see [here](https://github.com/snimu/hlb-gpt)) and trained two models on the same dataset
(`model_a` and `model_b_original`).
I rebasined `model_b_original` towards `model_a` to get `model_b_rebasin`, 
and interpolated between the resulting models in the following combinations:

- `model_a` and `model_b_original`
- `model_a` and `model_b_rebasin`
- `model_b_original` and `model_b_rebasin`

The results are shown below:

<p align="center">
  <img src="tests/results/hlb-gpt/images/hlb_line.png" alt="hlb-gpt" width="500"/>
</p>

Two things become obvious:

1. The rebasined model (`model_b_rebasin`) is way worse 
    than the original model (`model_b_original`)
2. Interpolation between any of the three models is anything but smooth.
    However, the interpolated models aren't significantly worse than the original ones;
    in many cases, they have similar performance

Point 2 makes me wonder which is better: training a model for a long time,
or training two models for a short time, rebasining one towards the other,
and retraining it for a short time.

However, in total, this experiment was a failure.

It will be interesting to see whether bigger models will show better results.
According to Ainsworth et al., the method should work better if the filters are bigger.
The `SpeedyLangNet` used here has ~30M parameters; the `vit_b_16` used in the next section
has ~86M parameters.

---

---

### torchvision.models

Below, I discuss results for models from `torchvision.models` 
that have two distinct sets of weights.

I rebasin the worse weights towards the better ones, then interpolate as in [HLB-GPT](#hlb-gpt).

I call the better of the original weights `model_a`, and the worse `model_b_original`.
The rebasined version of `model_b_original` is called `model_b_rebasin`.

---

#### Caveats

**Caveat 1**: I tested on CIFAR10, even though the models are trained on ImageNet.
This is because I don't currently have access to the ImageNet dataset as used 
for training the torchvision models
(the dataset should be [here](https://image-net.org/challenges/LSVRC/2012/), 
but seems unavailable. Please correct me if I'm wrong).
I will try to gain that access and repeat the experiments.
For now, these results have to suffice; I think that they are still interesting.

**Caveat 2**: For many of the models with BatchNorm, I did not recalculate the 
BatchNorm statistics before or after rebasing, even though it is recommended
(and a good idea to do when facing a new dataset). 
I also only used 10% of the evaluation dataset for evaluation.
Both of these things were done to speed up the experiments,
because I cannot afford to rent an A100 for a week.
**Those models are marked with a \*.**

**Caveat 3**: I did not include the results for all models below,
so if you want to see more, look at the files in 
[tests/results/torchvision/cifar10](tests/results/torchvision/cifar10)
and [tests/results/torchvision/cifar10/images](tests/results/torchvision/cifar10/images).

---

#### General takeaways

1. The weights that are better on ImageNet are also better on CIFAR10.
2. The rebasined model performs better than the original model. It would be interesting to see
   whether this is true if the better weights are rebasined towards the worse weights, 
   instead of the other way around.
3. The loss basins of the different models trained on ImageNet seem to 
   roughly surround a single loss basin for CIFAR10.
   - This means that the interpolated models tend to perform better than the original models.
   - The models interpolated between model_a and model_b_rebasin are usually the best.
   - This means that rebasing + interpolating can be a decent preparation for transfer learning, 
     especially as it is very fast.
   - There are no (significant) loss barriers between the original two models, 
     making this test less useful for testing the weight-matching method.
  
---

#### vit_b_16

| Key                | Weights                                                | 
|--------------------|--------------------------------------------------------|
| `model_a`          | `ViT_B_16_Weights.IMAGENET1K_V1`                       |
| `model_b_original` | `ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1 (orig)`    |
| `model_b_rebasin`  | `ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1 (rebasin)` |

Link: [ViT_B_16_Weights](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html)

Comparing the losses of the original models and the rebasined model, 
we can see that [takeaways 1 and 2](#general-takeaways) are true:

<p align="center">
  <img src="tests/results/torchvision/cifar10/images/vit_b_16_bar.png" alt="vit_b_16_bar" width="500"/>
</p>


When the losses of all models, including the interpolated ones, are drawn as below,
we can see that the plots support [all three takeaways](#general-takeaways):

<p align="center">
  <img src="tests/results/torchvision/cifar10/images/vit_b_16_line.png" alt="vit_b_16_line" width="500"/>
</p>


It seems that rebasing works very well for this model, which is not surprising.
Ainsworth et al. mention that their method works better if the filters are larger,
and the ViT models have very large filters.

Again, testing on ImageNet is crucial here! I will attempt to do so in the future.

---

#### wide_resnet50_2

| Key                | Weight                                            |
|--------------------|---------------------------------------------------|
| `model_a`          | `Wide_ResNet50_2_Weights.IMAGENET1K_V2`           |
| `model_b_original` | `Wide_ResNet50_2_Weights.IMAGENET1K_V1 (orig)`    |
| `model_b_rebasin`  | `Wide_ResNet50_2_Weights.IMAGENET1K_V1 (rebasin)` |

Link: [Wide_ResNet50_2_Weights](https://pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html)


This is a model with `BatchNorm`s in it. In this case, the `BatchNorm` statistics
were recalculated for every single model that was evaluated. For evaluation,
the full evaluation dataset was used.

The results are very similar to those of the other models.

<p align="center">
  <img 
    src="tests/results/torchvision/cifar10/images/wide_resnet50_2_bar.png" 
    alt="wide_resnet50_2_bar" 
    width="500"
  />
</p>

Interestingly, the best model is found by interpolating between the original models,
instead of `model_a` and `model_b_rebasin`, despite the fact that the latter 
is better than `model_b_original`!

<p align="center">
  <img src="tests/results/torchvision/cifar10/images/wide_resnet50_2_line.png" alt="wide_resnet50_2_line" width="500"/>
</p>

---

#### efficientnet_b1*

| Key                | Weight                                            |
|--------------------|---------------------------------------------------|
| `model_a`          | `EfficientNet_B1_Weights.IMAGENET1K_V2`           |
| `model_b_original` | `EfficientNet_B1_Weights.IMAGENET1K_V1 (orig)`    |
| `model_b_rebasin`  | `EfficientNet_B1_Weights.IMAGENET1K_V1 (rebasin)` |

Link: [EfficientNet_B1_Weights](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b1.html)

From both the losses of the original models and the rebasined model,
as well as the losses of the interpolated models, we can see that
[all takeaways](#general-takeaways) are true, except that the 
rebasined model lies very close to the optimum for CIFAR10 
(along the line of interpolation, which still leaves room for improvement):

<p align="center">
  <img src="tests/results/torchvision/cifar10/images/efficientnet_b1_bar.png" alt="efficientnet_b1_bar" width="500"/>
</p>


<p align="center">
  <img src="tests/results/torchvision/cifar10/images/efficientnet_b1_line.png" alt="efficientnet_b1_line" width="500"/>
</p>

---


## Acknowledgements

**Git Re-Basin:**

```
Ainsworth, Samuel K., Jonathan Hayase, and Siddhartha Srinivasa. 
"Git re-basin: Merging models modulo permutation symmetries." 
arXiv preprint arXiv:2209.04836 (2022).
```

Link: https://arxiv.org/abs/2209.04836 (accessed on April 9th, 2023)


**ImageNet:**

I've used the ImageNet Data from the 2012 ILSVRC competition.

```
Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, 
Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, 
Alexander C. Berg and Li Fei-Fei. (* = equal contribution) 
ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014
```

[Paper (link)](https://arxiv.org/abs/1409.0575) (Accessed on April 12th, 2023)


**Models:**

- The torchvision models (v.015), of course: https://pytorch.org/vision/0.15/models.html
- HLB-GPT by @tysam-code: https://github.com/tysam-code/hlb-gpt


**Other**

My code took inspiration from the following sources:

- https://github.com/themrzmaster/git-re-basin-pytorch

I used the amazing library `torchview` to visualize the models:

- https://github.com/mert-kurttutan/torchview

