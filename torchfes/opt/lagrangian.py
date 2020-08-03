from typing import Dict
from torch import nn, Tensor
from .. import properties as p
from .generalize import specialize_pos


class ConstraintLagrangian(nn.Module):
    def __init__(self, adj, eng, con):
        super().__init__()
        self.adj = adj
        self.eng = eng
        self.con = con

    def forward(self, inp: Dict[str, Tensor]):
        out = self.adj(inp)
        out = self.eng(out)
        con = self.con(out) - out[p.con_cen]
        L = out[p.eng] - (out[p.con_mul] * con).sum(1)
        if p.con_aug in out:
            L = L + (con.pow(2).sum(1) * 0.5 / out[p.con_aug])
        out[p.gen_eng] = L
        return out


class EnergyFunction(nn.Module):
    def __init__(self, adj, eng):
        super().__init__()
        self.adj = adj
        self.eng = eng

    def forward(self, inp: Dict[str, Tensor]):
        out = self.adj(inp)
        out = self.eng(out)
        out[p.gen_eng] = out[p.eng]
        return out
