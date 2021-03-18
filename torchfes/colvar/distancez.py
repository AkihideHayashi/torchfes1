from math import inf
from typing import Dict, List
import torch
from torch import nn, Tensor
from .. import properties as p


def comz(pos, mas, msk, dim):
    pos_mas = pos[:, :, dim] * mas[:, :]
    pos_mas.masked_fill_(~msk, 0.0)
    num = pos_mas.sum(dim=1)
    den = mas.masked_fill(~msk, 0.0).sum(dim=1)
    return num / den


class SlabDistanceZ(nn.Module):
    elm: Tensor
    pbc: Tensor
    slab: Tensor
    mol: Tensor

    def __init__(self, numel: int, slab: List[int], mol: List[int], wz: float,
                 dim: int = 2):
        super().__init__()
        elm = -torch.ones([numel, numel], dtype=torch.long)
        self.a = 0.5 / (wz * wz)
        self.register_buffer('elm', elm)
        self.register_buffer('slab', torch.tensor(slab))
        self.register_buffer('mol', torch.tensor(mol))
        self.register_buffer('pbc', torch.full([1], inf))
        self.dim = dim

    def forward(self, inp: Dict[str, Tensor]):
        pos = inp[p.pos]
        elm = inp[p.elm]
        mas = inp[p.mas]
        msk_slab = (elm[:, :, None] == self.slab[None, None, :]).any(dim=2)
        msk_mol = (elm[:, :, None] == self.mol[None, None, :]).any(dim=2)
        mol_z = comz(pos, mas, msk_mol, self.dim)
        slab_z = softmax(pos, msk_slab, self.dim, self.a)
        return (mol_z - slab_z)[:, None]


def softmax(pos, msk, dim, a):
    z = pos[:, :, dim].masked_fill(~msk, -inf)
    z_max, _ = z.detach().max(dim=1)
    dz = z - z_max
    w = torch.exp(dz * a)
    return (dz * w).masked_fill(~msk, 0.0).sum(dim=1) / w.sum(dim=1) + z_max
