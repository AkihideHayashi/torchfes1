from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p


def fnorm(x: Tensor, dim: int = 1):
    return x.flatten(dim).norm(2, dim)


class FIRE(nn.Module):
    def __init__(self, a0, n_min, f_a, f_inc, f_dec, dtm_max):
        super().__init__()
        self.a0 = a0
        self.n_min = n_min
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.f_a = f_a
        self.dtm_max = dtm_max
        self.a = torch.tensor([])
        self.count = torch.tensor([])

    def reset(self, inp: Dict[str, Tensor]):
        self.a = torch.ones_like(inp[p.eng_tot]) * self.a0
        self.count = torch.zeros_like(self.a)

    def forward(self, inp: Dict[str, Tensor]):
        if self.a.size() != inp[p.eng_tot].size():
            self.reset(inp)
        dtm = inp[p.dtm]
        vel = inp[p.mom] / inp[p.mas][:, :, None]
        frc = inp[p.frc]
        P = (vel * frc).sum(-1).sum(-1)
        ratio = (fnorm(vel) / fnorm(frc))[:, None, None]
        a = self.a[:, None, None]
        vel_new = (1 - a) * vel + a * frc * ratio
        positive = (P > 0) & (self.count > self.n_min)
        self.count += 1
        negative = P <= 0
        dtm[positive] *= self.f_inc
        dtm.clamp_max_(self.dtm_max)
        self.a[positive] *= self.f_a
        self.count[negative] = 0
        dtm[negative] *= self.f_dec
        vel_new[negative] = 0
        self.a[negative] = self.a0
        out = inp.copy()
        out[p.mom] = vel_new * out[p.mas][:, :, None]
        out[p.dtm] = dtm
        return out
