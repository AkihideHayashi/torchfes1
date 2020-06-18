from typing import Dict

import torch
from torch import Tensor, nn

import pointneighbor as pn
from pointneighbor import AdjSftSpc

from . import properties as p


class QuadraticRestraints(nn.Module):
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

    def forward(self, inp: Dict[str, Tensor], adj: AdjSftSpc):
        col: Tensor = self.col(inp, adj)
        # col.size() == (n_bch, n_col)
        assert col.dim() == 2
        eff = (col.sign() * self.sgn) >= 0
        res = (col * col * self.k * eff).sum(1)
        return res


class ClosePenalty(nn.Module):
    k: Tensor
    radius: Tensor

    def __init__(self, radius, k):
        super().__init__()
        self.register_buffer('k', k)
        self.register_buffer('radius', radius)

    def forward(self, inp: Dict[str, Tensor], adj: AdjSftSpc):
        rc = self.radius.max().item() * 2
        adj, vec_sod = pn.coo2_adj_vec_sod(adj, inp[p.pos], inp[p.cel], rc)
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
        return eng_mol
