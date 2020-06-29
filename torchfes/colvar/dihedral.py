import math
from typing import Optional, Dict
import torch
from torch import Tensor, nn
from .. import properties as p


def _dot(a: Tensor, b: Tensor):
    return (a * b).sum(dim=-1).unsqueeze(-1)


def _power(a: Tensor):
    return _dot(a, a)


def _projected_parallel(vec: Tensor, base: Tensor):
    return _dot(vec, base) / _dot(base, base) * base


def dihedral_inner(axis: Tensor, vec1: Tensor, vec2: Tensor):
    vec1_para = _projected_parallel(vec1, axis)
    vec1_perp = vec1 - vec1_para
    vec2_para = _projected_parallel(vec2, axis)
    vec2_perp = vec2 - vec2_para
    s_ = -torch.stack([vec1_perp, vec2_perp, axis], dim=-2).det().sign()
    c = (
        _dot(vec1_perp, vec2_perp) /
        (_power(vec1_perp) * _power(vec2_perp)).sqrt()
    ).squeeze(-1) * 0.99
    s = torch.where(s_ == 0, torch.ones_like(s_), s_).detach()
    return s * c.acos()


def dihedral(idx: Tensor, pos: Tensor, cel_inv: Optional[Tensor] = None):
    if cel_inv is not None:
        raise NotImplementedError()
    x = pos[:, idx, :]
    a = x[:, 0, :, :]
    b = x[:, 1, :, :]
    c = x[:, 2, :, :]
    d = x[:, 3, :, :]
    axis = c - b
    vec1 = a - b
    vec2 = d - c
    return dihedral_inner(axis, vec1, vec2)


class Dihedral(nn.Module):
    def __init__(self, idx, cel=False):
        super().__init__()
        self.idx = idx
        self.cel = cel
        self.pbc = torch.tensor([2 * math.pi for _ in range(idx.size(1))])
        assert not self.cel

    def forward(self, inp: Dict[str, Tensor]):
        return dihedral(self.idx, inp[p.pos])
