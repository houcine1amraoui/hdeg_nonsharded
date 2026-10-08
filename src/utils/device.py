import importlib

import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        xm = importlib.import_module("torch_xla.core.xla_model")
        return xm.xla_device()
    except (ImportError, ModuleNotFoundError):
        return torch.device("cpu")