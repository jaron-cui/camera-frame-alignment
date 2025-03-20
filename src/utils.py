import os

import einops
import typing

import numpy as np
import torch


PathLike = typing.Union[str, os.PathLike]


def cache_operation(operation: typing.Callable[[], torch.Tensor], cache_path: PathLike = None):
    if cache_path:
        try:
            return torch.load(cache_path, weights_only=True)
        except FileNotFoundError:
            pass
    data = operation()
    if cache_path:
        torch.save(data, cache_path)
    return data


def tensor_to_image(x: torch.Tensor) -> np.ndarray:
    x = (x.detach().cpu() * 255).type(torch.uint8)
    x = einops.rearrange(x, 'c h w -> h w c')
    x = x.numpy()
    return x


def tensors_to_image(x: torch.Tensor) -> np.ndarray:
    x = (x.detach().cpu() * 255).type(torch.uint8)
    x = einops.rearrange(x, 't b c h w -> t b h w c')
    x = x.numpy()
    return x
