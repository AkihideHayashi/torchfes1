import math
from typing import Dict, Union, List
import torch
from torch import nn, Tensor
from .. import properties as p


def fix_msk(mol: Dict[str, Tensor], idx: Tensor):
    _, atm, dim = mol[p.pos].size()
    msk = torch.zeros([atm, dim], dtype=torch.bool, device=idx.device)
    msk[idx, :] = True
    return msk


class Fix(nn.Module):
    idx: Tensor

    def __init__(self, idx: Union[Tensor, List[int]]):
        super().__init__()
        if isinstance(idx, list):
            idx = torch.tensor(idx)
        self.register_buffer('idx', idx)

    def forward(self, mol: Dict[str, Tensor]):
        out = mol.copy()
        msk = fix_msk(mol, self.idx)[None, :, :]
        if p.fix_msk not in out:
            out[p.fix_msk] = msk
        else:
            out[p.fix_msk] = out[p.fix_msk] | msk
        return out


class FixGen(nn.Module):
    pbc: Tensor
    idx: Tensor

    def __init__(self, idx: Union[Tensor, List[int]], num_dim: int):
        super().__init__()
        if isinstance(idx, list):
            idx = torch.tensor(idx, dtype=torch.long)
        n = idx.numel() * num_dim
        self.register_buffer('idx', idx)
        self.register_buffer('pbc', torch.ones(n) * math.inf)

    def forward(self, mol: Dict[str, Tensor]):
        msk = fix_msk(mol, self.idx)
        return mol[p.pos][:, msk]
