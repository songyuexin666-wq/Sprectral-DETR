# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

"""
onnx optimizer and symbolic registry
"""
from . import optimizer
from . import symbolic

from .optimizer import OnnxOptimizer
from .symbolic import CustomOpSymbolicRegistry
