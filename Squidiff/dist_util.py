"""
Helpers for Accelerate-backed multi-GPU training and checkpoint I/O.
"""

import os

import torch as th
from accelerate import Accelerator


_ACCELERATOR = None
_ACCELERATOR_CONFIG = None


def setup_accelerator(*, use_fp16=False):
    """
    Create a singleton Accelerator instance for the current process.
    """
    global _ACCELERATOR, _ACCELERATOR_CONFIG

    mixed_precision = "fp16" if use_fp16 else "no"
    requested_config = {"mixed_precision": mixed_precision}

    if _ACCELERATOR is None:
        _ACCELERATOR = Accelerator(**requested_config)
        _ACCELERATOR_CONFIG = requested_config
    elif _ACCELERATOR_CONFIG != requested_config:
        raise RuntimeError(
            "Accelerator has already been initialized with a different "
            f"configuration: {_ACCELERATOR_CONFIG} != {requested_config}"
        )

    return _ACCELERATOR


def accelerator():
    """
    Return the shared Accelerator instance, creating a default one if needed.
    """
    if _ACCELERATOR is None:
        return setup_accelerator()
    return _ACCELERATOR


def dev():
    """
    Get the device managed by Accelerate.
    """
    return accelerator().device


def is_main_process():
    return accelerator().is_main_process


def num_processes():
    return accelerator().num_processes


def process_index():
    return accelerator().process_index


def prepare(*objects):
    return accelerator().prepare(*objects)


def unwrap_model(model):
    return accelerator().unwrap_model(model)


def backward(loss):
    accelerator().backward(loss)


def wait_for_everyone():
    accelerator().wait_for_everyone()


def reduce_tensor(tensor, reduction="mean"):
    return accelerator().reduce(tensor, reduction=reduction)


def gather_for_metrics(tensor):
    return accelerator().gather_for_metrics(tensor)


def pad_across_processes(tensor, dim=0, pad_index=0):
    return accelerator().pad_across_processes(tensor, dim=dim, pad_index=pad_index)


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    accelerator().save(obj, path)


def load_state_dict(path, map_location="cpu", **kwargs):
    """
    Load a PyTorch checkpoint from local storage.
    """
    state_dict = th.load(path, map_location=map_location, **kwargs)

    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            return state_dict["state_dict"]
        if "model" in state_dict:
            return state_dict["model"]

    return state_dict


def sync_params(_params):
    """
    Retained for compatibility with older call sites.
    Accelerate handles parameter synchronization when wrapping the model.
    """
    return
