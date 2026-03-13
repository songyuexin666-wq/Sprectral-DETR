# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

"""
CustomOpSymbolicRegistry class
"""
from copy import deepcopy

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_helper import _get_tensor_dim_size, _get_tensor_sizes
from torch.autograd import Function


class CustomOpSymbolicRegistry:
    # _SYMBOLICS = {}
    _OPTIMIZER = []

    @classmethod
    def optimizer(cls, fn):
        cls._OPTIMIZER.append(fn)


def register_optimizer():
    def optimizer_wrapper(fn):
        CustomOpSymbolicRegistry.optimizer(fn)
        return fn
    return optimizer_wrapper
