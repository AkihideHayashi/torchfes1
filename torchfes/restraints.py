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


class Repulsive(nn.Module):
    eng_atm: Tensor
    radius: Tensor

    def __init__(self):
        super().__init__()
        self.rc = model.radius.max().item()
        self.register_buffer('radius', model.radius)
        self.adj = fes.adj.SetAdjSftSpcVecSod(
            pn.Coo2FulSimple(self.rc), [(p.coo, self.rc)]
        )
        self.register_buffer('eng_atm',
                             torch.tensor([single[o] for o in model.order]))

    def forward(self, inp: Dict[str, Tensor]):
        eng_max = self.eng_atm[inp[p.elm]].sum(-1)
        eng_big = inp[p.eng] >= eng_max

        out = self.adj(inp)
        adj: pn.AdjSftSpc = fes.adj.get_adj_sft_spc(out, p.coo, self.rc)
        vec, sod = pn.coo2_vec_sod(adj, out[p.pos], out[p.cel])
        n, i, j = pn.coo2_n_i_j(adj)
        elm = inp[p.elm]
        ei = elm[n, i]
        ej = elm[n, j]
        too_close_ = (self.radius[ei] + self.radius[ej]).pow(2) >= sod
        too_close = torch.scatter_add(
            torch.zeros_like(out[p.eng], dtype=torch.long), 0,
            n[too_close_], torch.ones_like(n[too_close_])) > 0
        return eng_big | too_close

