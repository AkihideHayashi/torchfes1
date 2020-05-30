from typing import Union, List
import numpy as np
import torch
from torch import Tensor


def pad_torch(tensor: Tensor, size: List[int], value: float):
    pad_: List[int] = []
    for i, s in enumerate(size):
        dif = max(0, s - tensor.size(i))
        pad_.append(dif)
        pad_.append(0)
    pad_.reverse()
    return torch.nn.functional.pad(tensor, pad_,
                                   mode='constant', value=value)


def pad_cat_torch(tensors: List[Tensor], value: float, dim: int = 0):
    t0 = tensors[0]
    for tensor in tensors:
        assert t0.dim() == tensor.dim()
    size: List[int] = []
    for i in range(t0.dim()):
        size.append(max([tensor.size(i) for tensor in tensors]))
    size[dim] = 0
    padded = [pad_torch(tensor, size, value) for tensor in tensors]
    return torch.cat(padded, dim=dim)


def pad_numpy(array: np.ndarray, shape: List[int],
              constant_values: Union[bool, int, float, str]):
    assert array.ndim == len(shape)
    pad_width = [(0, max(0, si - asi)) for si, asi in zip(shape, array.shape)]
    return np.pad(array, pad_width, mode='constant',
                  constant_values=constant_values)


def pad_cat_numpy(arrays: List[np.ndarray],
                  constant_values: Union[bool, int, float, str], axis=0):
    for arr in arrays:
        assert arr.ndim == arrays[0].ndim
    shape = []
    for i in range(arrays[0].ndim):
        shape.append(max(arr.shape[i] for arr in arrays))
    shape[axis] = 0
    padded = [pad_numpy(arr, shape, constant_values) for arr in arrays]
    return np.concatenate(padded, axis=axis)


def pad(tensor, size, value):
    if isinstance(tensor, Tensor):
        return pad_torch(tensor, size, value)
    elif isinstance(tensor, np.ndarray):
        return pad_numpy(tensor, size, value)
    else:
        raise RuntimeError()


def pad_cat(tensors, value, dim=0):
    t0 = tensors[0]
    if isinstance(t0, Tensor):
        return pad_cat_torch(tensors, value, dim)
    elif isinstance(t0, np.ndarray):
        return pad_cat_numpy(tensors, value, dim)
    else:
        raise RuntimeError()
