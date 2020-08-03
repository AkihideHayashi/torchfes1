from typing import Dict, List

import torch
from torch import Tensor, nn

import torchfes as fes

from . import properties as p


def add_eng_res(mol: Dict[str, Tensor], eng: Tensor):
    mol = mol.copy()
    mol[p.eng_res] = torch.cat([mol[p.eng_res], eng], dim=1)
    return mol


class Restraints(nn.Module):
    def __init__(self, restraints: List[nn.Module]):
        super().__init__()
        self.restraints = nn.ModuleList(restraints)

    def forward(self, mol: Dict[str, Tensor]):
        pos = mol[p.pos]
        n_bch = pos.size(0)
        mol = mol.copy()
        mol[p.eng_res] = torch.zeros([n_bch, 0],
                                     dtype=pos.dtype, device=pos.device)
        for restraint in self.restraints:
            mol = restraint(mol)
        return mol


class HarmonicRestraints(nn.Module):
    msk: Tensor

    def __init__(self, msk, sgn, k):
        """
        Args:
            pot: potential function.
            col: colvar function.
            sgn: signature for each colvar.
                 -1 -> restraint for negative value.
                 0 -> restraint for all value.
                 1 -> restraint for positive value.
            k: spring constant.
        """
        super().__init__()
        self.register_buffer('msk', msk)
        self.sgn = sgn
        self.k = k

    def forward(self, inp: Dict[str, Tensor]):
        msk = self.msk
        col: Tensor = (inp[p.col_var] - inp[p.col_cen])[:, msk]
        # col.size() == (n_bch, n_col)
        assert col.dim() == 2
        assert col.size(0) == inp[p.pos].size(0)
        eff = (col.sign() * self.sgn) >= 0
        res = (col * col * self.k * eff)
        return add_eng_res(inp, res)


class ClosePenalty(nn.Module):
    k: Tensor
    radius: Tensor

    def __init__(self, radius, k):
        super().__init__()
        self.register_buffer('k', k)
        self.register_buffer('radius', radius)
        self.rc = self.radius.max().item() * 2

    def forward(self, inp: Dict[str, Tensor]):
        adj = fes.adj.get_adj_sft_spc(inp, p.coo, self.rc)
        vec_sod = fes.adj.vec_sod(inp, adj)
        n, i, j, _ = adj.adj.unbind(0)
        ei = inp[p.elm][n, i]
        ej = inp[p.elm][n, j]
        sod = vec_sod.sod
        dis = sod.sqrt()
        k = self.k[ei] + self.k[ej]
        R = self.radius[ei] + self.radius[ej]
        mask = dis < R
        eng_bnd = k * (dis - R).pow(2) * mask
        n_bch, n_atm, _ = adj.spc.size()
        eng_atm = torch.index_add(
            torch.zeros((n_bch * n_atm)).to(inp[p.pos]),
            0, n * n_atm + i, eng_bnd
        ).view((n_bch, n_atm))
        eng_mol = eng_atm.sum(1)
        return add_eng_res(inp, eng_mol.unsqueeze(1))
