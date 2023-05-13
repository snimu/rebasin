from __future__ import annotations

from typing import Union

from torchview import FunctionNode, ModuleNode, TensorNode

NODE_TYPES = Union[ModuleNode, TensorNode, FunctionNode]
