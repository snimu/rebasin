# rebasin
An implementation of methods described in 
["Git Re-basin"-paper by Ainsworth et al.](https://arxiv.org/abs/2209.04836)

Can be applied to arbitrary models, without modification.

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Results: Weight-matching (PermutationCoordinateDescent)](#results-weight-matching-permutationcoordinatedescent)
  - [General takeaways](#general-takeaways)
  - [vit_b_16](#vitb16)
  - [efficientnet_b1](#efficientnetb1)
- [Plans](#plans)

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

## Results: Weight-matching (PermutationCoordinateDescent)

Below, I present some preliminary results. In them, I used `torchvision`-models with two different 
pre-trained weights. I interpolated between the two sets of weights, 
"rebasined" one, and interpolated again, saving all the losses.

**Caveat 1**: I tested on CIFAR10, even though the models are trained on ImageNet.
This is because I don't currently have access to the ImageNet dataset as used 
for training the torchvision models
(the dataset should be [here](https://image-net.org/challenges/LSVRC/2012/), 
but seems unavailable. Please correct me if I'm wrong).
I will try to gain that access and repeat the experiments.
For now, these results have to suffice; I think that they are still interesting.

**Caveat 2**: For the models with BatchNorm, I did not recalculate the 
BatchNorm statistics before or after rebasing, even though it is recommended
(and a good idea to do when facing a new dataset). 
I also only used 10% of the evaluation dataset for evaluation.
Both of these things were done to speed up the experiments,
because I cannot afford to rent an A100 for a week.
**Those models are marked with a \*.**

**Caveat 3**: I did not include the results for all models below,
so if you want to see more, look at the files in [tests/results/](tests/results/)
and [tests/results/images](tests/results/images).


### General takeaways

1. The weights that are better on ImageNet are also better on CIFAR10.
2. The rebasined model performs better than the original model.
3. The loss basins of the different models trained on ImageNet seem to 
   roughly surround a single loss basin for CIFAR10.
   - This means that the interpolated models tend to perform better than the original models.
   - The models interpolated between model_a and model_b_rebasin are usually the best.
   - This means that rebasing + interpolating can be a decent preparation for transfer learning, 
     especially as it is very fast.
   - There are no (significant) loss barriers between the original two models, 
     making this test less useful for testing the weight-matching method.
  

### vit_b_16

Comparing the losses of the original models and the rebasined model, 
we can see that [takeaways 1 and 2](#general-takeaways) are true:

<p align="center">
  <img src="tests/results/images/vit_b_16_bar.png" alt="vit_b_16_bar" width="500"/>
</p>


When the losses of all models, including the interpolated ones, are drawn as below,
we can see that the plots support [all three takeaways](#general-takeaways):

<p align="center">
  <img src="tests/results/images/vit_b_16_line.png" alt="vit_b_16_line" width="500"/>
</p>


It seems that rebasing works very well for this model, which is not surprising.
Ainsworth et al. mention that their method works better if the filters are larger,
and the ViT models have very large filters.

Again, testing on ImageNet is crucial here! I will attempt to do so in the future.


### efficientnet_b1*

From both the losses of the original models and the rebasined model,
as well as the losses of the interpolated models, we can see that
[all takeaways](#general-takeaways) are true, except that the 
rebasined model lies very close to the optimum for CIFAR10 
(along the line of interpolation, which still leaves room for improvement):

<p align="center">
  <img src="tests/results/images/efficientnet_b1_bar.png" alt="efficientnet_b1_bar" width="500"/>
</p>


<p align="center">
  <img src="tests/results/images/efficientnet_b1_line.png" alt="efficientnet_b1_line" width="500"/>
</p>


## Plans

Here, I present my near-term plans for this package. They may change.

- [x] Implement weight-matching
- [x] Implement linear interpolation
- [ ] Increase unittest-coverage and test on push / merge (GitHub Actions)
- [ ] Test on ImageNet
- [ ] Test on other datasets with other models &mdash; for example, I would like to test
      `rebasin` on [hlb-gpt](https://github.com/tysam-code/hlb-gpt)
- [ ] Create proper documentation and write docstrings
- [ ] Implement other rebasing methods:
    1. Straight-through estimator. This allegedly has better results than weight-matching,
       though at higher computational cost.
    2. Activation-matching. Just for completeness.
- [ ] Implement other interpolation methods:
    1. Quadratic interpolation
    2. Cubic interpolation
    3. Spline interpolation
      
    This is especially relevant for interpolating between more than two models at a time.
