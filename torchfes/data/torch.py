from typing import List
import torch
from torch import Tensor


def _size_max(a: List[int], b: List[int]):
    if a:
        if b:
            c = []
            assert len(a) == len(b)
            for ai, bi in zip(a, b):
                c.append(max(ai, bi))
            return c
        else:
            return a
    else:
        if b:
            return b
        else:
            return[]


def size_max(tensors: List[Tensor]):
    size: List[int] = []
    for tensor in tensors:
        size = _size_max(size, list(tensor.size()))
    return size


def pad(tensors: List[Tensor], value: float, dim: int = 0):
    size = size_max(tensors)
    size[dim] = 0
    padded = [_pad_siz(tensor, size, value) for tensor in tensors]
    return padded


def cat(tensors: List[Tensor], value: float, dim: int = 0):
    t0 = tensors[0]
    for tensor in tensors:
        assert t0.dim() == tensor.dim()
    size: List[int] = []
    for i in range(t0.dim()):
        size.append(max([tensor.size(i) for tensor in tensors]))
    size[dim] = 0
    padded = [_pad_siz(tensor, size, value) for tensor in tensors]
    return torch.cat(padded, dim=dim)


def stack(tensors: List[Tensor], value: float, dim: int = 0):
    t0 = tensors[0]
    for tensor in tensors:
        assert t0.dim() == tensor.dim()
    if t0.size() == ():
        assert dim == 0
        return torch.tensor(tensors)
    size: List[int] = []
    for i in range(t0.dim()):
        size.append(max([tensor.size(i) for tensor in tensors]))
    padded = [_pad_siz(tensor, size, value) for tensor in tensors]
    return torch.stack(padded, dim=dim)


def _pad_siz(tensor: Tensor, size: List[int], value: float):
    pad_: List[int] = []
    for i, s in enumerate(size):
        dif = max(0, s - tensor.size(i))
        pad_.append(dif)
        pad_.append(0)
    pad_.reverse()
    return torch.nn.functional.pad(tensor, pad_,
                                   mode='constant', value=value)


def pad_dim_siz(tensor: Tensor, dim: int, siz: int, value: float):
    size = list(tensor.size())
    size[dim] = siz
    return _pad_siz(tensor, size, value)
