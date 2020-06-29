from typing import Dict, List

import torch
from torch import Tensor, nn

import torchfes as fes

from . import properties as p


class MultipleRestraints(nn.Module):
    def __init__(self, restraints: List[nn.Module]):
        super().__init__()
        self.restraints = restraints

    def forward(self, inp: Dict[str, Tensor]):
        eng = []
        for restraint in self.restraints:
            eng.append(restraint(inp))
        return torch.cat(eng, dim=1)


class HarmonicRestraints(nn.Module):
    def __init__(self, col, sgn, k):
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
        self.col = col
        self.sgn = sgn
        self.k = k

    def forward(self, inp: Dict[str, Tensor]):
        col: Tensor = self.col(inp) - inp[p.res_cen]
        # col.size() == (n_bch, n_col)
        assert col.dim() == 2
        assert col.size(0) == inp[p.pos].size(0)
        eff = (col.sign() * self.sgn) >= 0
        res = (col * col * self.k * eff)
        return res


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
        vec_sod = fes.adj.get_vec_sod(inp, p.coo, self.rc)
        n, i, j, _ = adj.adj.unbind(0)
        ei = inp[p.elm][n, i]
        ej = inp[p.elm][n, j]
        sod = vec_sod.sod
        dis = sod.sqrt()
        k = self.k[ei] + self.k[ej]
        R = self.radius[ei] + self.radius[ej]
        mask = dis < R
        if mask.any():
            print('ClosePenalty.')
        eng_bnd = k * (dis - R).pow(2) * mask
        n_bch, n_atm, _ = adj.spc.size()
        eng_atm = torch.index_add(
            torch.zeros((n_bch * n_atm)).to(inp[p.pos]),
            0, n * n_atm + i, eng_bnd
        ).view((n_bch, n_atm))
        eng_mol = eng_atm.sum(1)
        return eng_mol.unsqueeze(1)
