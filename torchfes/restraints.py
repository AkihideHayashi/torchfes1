import logging
from typing import Dict
import torch
from torch import nn, Tensor
from . import properties as p
from .adj import Adjacent, AdjSftSizVecSod

_logger = logging.getLogger(__name__)


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

    def forward(self, inp: Dict[str, Tensor]):
        col: Tensor = self.col(inp)
        assert col.dim() == 2
        eff = (col.sign() * self.sgn) >= 0
        res = (col * col * self.k * eff).sum(1)
        return res


class ClosePenalty(nn.Module):
    def __init__(self, adj: Adjacent, radius, k):
        super().__init__()
        self.radius = radius
        self.k = k
        self.adj = adj

    def forward(self, inp: Dict[str, Tensor]):
        adj: AdjSftSizVecSod = self.adj(inp)
        n, i, j, _ = adj.adj.unbind(0)
        ei = inp[p.elm][n, i]
        ej = inp[p.elm][n, j]
        sod = adj.sod
        dis = sod.sqrt()
        k = self.k[ei] + self.k[ej]
        R = self.radius[ei] + self.radius[ej]
        mask = dis < R
        if mask.any():
            _logger.warning('ClosePenalty.')
        eng_bnd = k * (dis - R).pow(2) * mask
        n_bch, n_atm = adj.siz
        eng_atm = torch.index_add(
            torch.zeros((n_bch * n_atm)).to(inp[p.pos]),
            0, n * n_atm + i, eng_bnd
        ).view((n_bch, n_atm))
        eng_mol = eng_atm.sum(1)
        return eng_mol
