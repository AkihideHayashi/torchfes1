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
        a0: \alpha_start. 0 < a0 < 1
        n_min: N_min. 0 < n_min
        f_a: f_\alpha. 0 < f_a < 1
        f_inc: f_inc. 1 < f_inc
        f_dec: f_dec. 0 < f_dec < 1
        dtm_max: \Delta t_max. 0 < dtm_max
    """
    def __init__(self, dtm_max: float, a0: float = 0.1, n_min: int = 5,
                 f_a: float = 0.99, f_inc: float = 1.1, f_dec: float = 0.5):
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

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        if p.fir_alp not in out:
            out[p.fir_alp] = torch.ones_like(out[p.eng]) * self.a0
        if p.fir_cnt not in out:
            out[p.fir_cnt] = torch.zeros_like(out[p.fir_alp])
        dtm = out[p.dtm]
        vel = out[p.mom] / out[p.mas][:, :, None]
        frc = out[p.frc]
        P = (vel * frc).sum(-1).sum(-1)
        ratio = (_fnorm(vel) / _fnorm(frc))[:, None, None]
        a = out[p.fir_alp][:, None, None]
        vel_new = (1 - a) * vel + a * frc * ratio
        positive = (P > 0) & (out[p.fir_cnt] > self.n_min)
        out[p.fir_cnt] += 1
        negative = P <= 0
        dtm[positive] *= self.f_inc
        dtm.clamp_max_(self.dtm_max)
        out[p.fir_alp][positive] *= self.f_a
        out[p.fir_cnt][negative] = torch.zeros_like(out[p.fir_cnt][negative])
        dtm[negative] *= self.f_dec
        vel_new[negative] *= torch.zeros_like(vel_new[negative])
        out[p.fir_alp][negative] = torch.ones_like(out[p.fir_alp]
                                                   )[negative] * self.a0
        out[p.mom] = vel_new * out[p.mas][:, :, None]
        out[p.dtm] = dtm
        return out
