from typing import Dict
from torch import nn, Tensor
from .. import properties as p


class SD(nn.Module):
    def __init__(self, evl: nn.Module, lmd: Tensor):
        super().__init__()
        self.lmd = lmd
        self.evl = evl

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        out[p.pos] = out[p.pos] + out[p.frc] * self.lmd[None, None, :]
        out = self.evl(out)
        return out
