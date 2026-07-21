# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Compatibility for pytorchvideo 0.1.5 with torchvision >= 0.13."""

import sys


def apply_torchvision_pytorchvideo_compat() -> None:
    """
    pytorchvideo imports ``torchvision.transforms.functional_tensor``, which was
    removed from public API in newer torchvision (logic lives in ``_functional_tensor``).
    Register the alias before any pytorchvideo import.
    """
    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ModuleNotFoundError:
        import torchvision.transforms._functional_tensor as functional_tensor

        sys.modules[name] = functional_tensor
