from typing import List
import torch
from torch import Tensor


def pad(tensor: Tensor, size: List[int], value: float):
    pad_: List[int] = []
    for i, s in enumerate(size):
        dif = max(0, s - tensor.size(i))
        pad_.append(dif)
        pad_.append(0)
    pad_.reverse()
    return torch.nn.functional.pad(tensor, pad_,
                                   mode='constant', value=value)


def cat(tensors: List[Tensor], value: float, dim: int = 0):
    t0 = tensors[0]
    for tensor in tensors:
        assert t0.dim() == tensor.dim()
    size: List[int] = []
    for i in range(t0.dim()):
        size.append(max([tensor.size(i) for tensor in tensors]))
    size[dim] = 0
    padded = [pad(tensor, size, value) for tensor in tensors]
    return torch.cat(padded, dim=dim)


def stack(tensors: List[Tensor], value: float, dim: int = 0):
    t0 = tensors[0]
    for tensor in tensors:
        assert t0.dim() == tensor.dim()
    size: List[int] = []
    for i in range(t0.dim()):
        size.append(max([tensor.size(i) for tensor in tensors]))
    padded = [pad(tensor, size, value) for tensor in tensors]
    return torch.stack(padded, dim=dim)
