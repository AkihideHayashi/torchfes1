from typing import Dict

import torch
from torch import Tensor, nn

from .. import properties as p


def _fnorm(x: Tensor, dim: int = 1):
    return x.flatten(dim).norm(2, dim)


class FIRE(nn.Module):
    r"""FIRE relaxation.
    DOI: 10.1103/PhysRevLett.97.170201
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.170201

    The following processes are performed between the steps of MD.

    P = F \cdot v
    v -> (1 - \alpha) v + \alpha F\hat |v|
    if P > 0 for N_min times in a row:
        \Delta t -> min(\Delta t f_inc, \Delta t_max)
        \alpha -> \alpha f_\alpha
    if P <= 0:
        \Delta t -> \Delta t f_dec
        v -> 0
        \alpha -> \alpha_start

    Args:
        a0: \alpha_start
        n_min: N_min
        f_a: f_\alpha
        f_inc: f_inc
        f_dec: f_dec
        dtm_max: \Delta t_max
    """
    def __init__(self, a0: float, n_min: int, f_a: float,
                 f_inc: float, f_dec: float, dtm_max: float):
        super().__init__()
        assert 0 < f_a < 1
        assert f_inc > 1
        assert 0 < f_dec < 1
        self.a0 = a0
        self.n_min = n_min
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.f_a = f_a
        self.dtm_max = dtm_max
        self.a = torch.tensor([])
        self.count = torch.tensor([])

    def reset(self, inp: Dict[str, Tensor]):
        self.a = torch.ones_like(inp[p.eng]) * self.a0
        self.count = torch.zeros_like(self.a)

    def forward(self, inp: Dict[str, Tensor]):
        if self.a.size() != inp[p.eng].size():
            self.reset(inp)
        dtm = inp[p.dtm]
        vel = inp[p.mom] / inp[p.mas][:, :, None]
        frc = inp[p.frc]
        P = (vel * frc).sum(-1).sum(-1)
        ratio = (_fnorm(vel) / _fnorm(frc))[:, None, None]
        a = self.a[:, None, None]
        vel_new = (1 - a) * vel + a * frc * ratio
        positive = (P > 0) & (self.count > self.n_min)
        self.count += 1
        negative = P <= 0
        dtm[positive] *= self.f_inc
        dtm.clamp_max_(self.dtm_max)
        self.a[positive] *= self.f_a
        self.count[negative] = torch.zeros_like(self.count[negative])
        dtm[negative] *= self.f_dec
        vel_new[negative] *= torch.zeros_like(vel_new[negative])
        self.a[negative] = torch.ones_like(
            negative, dtype=self.a.dtype) * self.a0
        out = inp.copy()
        out[p.mom] = vel_new * out[p.mas][:, :, None]
        out[p.dtm] = dtm
        return out
