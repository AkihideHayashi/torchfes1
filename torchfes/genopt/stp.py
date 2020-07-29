import torch
from torch import nn, Tensor
from ..general import PosEngFrc


class LogSmapler(nn.Module):
    """Simple line search sampler.
    if stp is too big, stp -> stp * f_inc
    if stp is too small, stp -> stp * f_dec
    """
    def __init__(self, f_inc: float, f_dec: float, a0=1.0):
        super().__init__()
        assert f_inc > 1
        assert 0 < f_dec < 1
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.stp = torch.tensor([])
        self.a0 = a0

    def init(self, pef: PosEngFrc, init: Tensor, reset: bool):
        if self.stp.size() != pef.eng.size():
            self.stp = torch.ones_like(pef.eng) * self.a0
        elif reset:
            new = torch.ones_like(init, dtype=self.stp.dtype) * self.a0
            self.stp[init] = new[init]
        return self.stp

    def peek(self):
        return self.stp

    def forward(self, con: Tensor, _: PosEngFrc, flt: Tensor):
        self.stp[flt & (con == 1)] *= self.f_dec
        self.stp[flt & (con == -1)] *= self.f_inc
        return self.stp
