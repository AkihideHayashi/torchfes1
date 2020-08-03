import math
from typing import Dict, List
import torch
from torch import nn, Tensor
from .. import properties as p


def fix(idx: List[int], num_dim: int = 3):
    return FixGen(Fix(torch.tensor(idx), num_dim))


def fix_msk(idx: Tensor, size: List[int]):
    msk = torch.zeros(size, dtype=torch.bool, device=idx.device)
    assert len(size) == 2
    msk[idx, :] = True
    return msk


class Fix(nn.Module):
    idx: Tensor

    def __init__(self, idx: Tensor, num_dim: int = 3):
        super().__init__()
        self.register_buffer('idx', idx)
        self.num_dim = num_dim

    def forward(self, mol: Dict[str, Tensor]):
        size = list(mol[p.pos].size())[1:]
        return fix_msk(self.idx, size)


class FixGen(nn.Module):
    pbc: Tensor

    def __init__(self, fix: Fix):
        super().__init__()
        self.fix = fix
        n = fix.idx.numel() * fix.num_dim
        self.register_buffer('pbc', torch.ones(n) * math.inf)

    def forward(self, mol: Dict[str, Tensor]):
        msk = self.fix(mol)
        return mol[p.pos][:, msk]
