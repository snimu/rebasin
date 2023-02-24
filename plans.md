## What do I need?

Here are some things that I need to achieve and first thoughts on how they might be achievable.

### I need a way to trace linear paths through the network.

Use *hooks* and *pre-hooks* to track the modules
(see [torchinfo](https://github.com/TylerYep/torchinfo)). 
Determine module order by tracking their in- and outputs. 
This might need a `inputs_dict: dict[torch.Tensor, list[torch.nn.Module]]`
and `outputs_dict: dict[torch.nn.Module, list[torch.Tensor]]`. 

This enables looping not just over the modules, but over the paths that the data takes:

```python
import torch


def rand_path_permutation(
        paths: list[list[torch.nn.Module]]
) -> list[list[torch.nn.Module]]:
    ...


def rand_module_permutation(path: list[torch.nn.Module]) -> list[torch.nn.Module]:
    ...


paths: list[list[torch.nn.Module]] = ...

converged = False

while not converged:
    for path in rand_path_permutation(paths):
        for module in path:
            ...  # Update permutations
```

This way, I can achieve permutation matrices that are optimally correlated to 
their immediate neighbors not only from one previous and one following module, but 
from all immediately previous and following modules, even if they run in parallel 
on the data produced by the module in question.

In other words, a module might have two modules consuming the data it produces. 
This is, for example, the case in Bottleneck-layers in ResNets. 
Optimizing through every path several times in random permutations 
allows for (hopefully) better permutation matrices.

Actually, the above code is not fully correct, because I want to split these paths 
into sub-paths / partial-paths that have only modules with broadcastable weights.
Why? Well it seems infeasible to me to develop permutations-matrices with the given 
mathematical formulations that reach over, for example, `torch.nn.MaxPool`-layers. 

### I need a better concept of convergence

Currently, I interpret convergence as no more change to the permutation matrices.
This might be ideal in purely linear models, but for models that branch
(as described above), this metric might never converge.

The alternative is using the convergence mechanism from 
[git-rebasin-pytorch](https://github.com/themrzmaster/git-re-basin-pytorch/blob/main/utils/weight_matching.py#L238).
Of course, I'll have to understand what it does, first, because I want to implement a 
more modular and general way of re-basin, which will necessarily differ in its implementation.
