from typing import List, Dict
import torch
from torch import nn, Tensor
from .. import properties as p


def _is_tensor(x):
    if isinstance(x, Tensor):
        return x
    else:
        raise RuntimeError()


def add_colvar(mol: Dict[str, Tensor], new: Tensor):
    out = mol.copy()
    out[p.col_var] = torch.cat([out[p.col_var], new], dim=1)
    return out


class ColVar(nn.Module):
    def __init__(self, colvars: List[nn.Module]):
        super().__init__()
        self.colvars = nn.ModuleList(colvars)
        pbc: List[Tensor] = [_is_tensor(col.pbc) for col in self.colvars]
        idx = [torch.ones(t.size()) * i for i, t in enumerate(pbc)]
        self.pbc = torch.cat(pbc)
        self.idx = torch.cat(idx)

    def __getitem__(self, keys: List[nn.Module]):
        num_ = []
        for key in keys:
            for i, m in enumerate(self.colvars):
                if m is key:
                    num_.append(i)
                    break
            else:
                raise RuntimeError()
        num = torch.tensor(num_)
        msk = (self.idx[:, None] == num[None, :]).any(1)
        return msk, self.pbc[msk]

    def forward(self, mol: Dict[str, Tensor]):
        pos = mol[p.pos]
        n_bch = pos.size(0)
        mol[p.col_var] = torch.zeros([n_bch, 0],
                                     dtype=pos.dtype, device=pos.device)
        for colvar in self.colvars:
            mol = colvar(mol)
        return mol
