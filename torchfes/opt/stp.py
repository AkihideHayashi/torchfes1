import torch
from torch import nn, Tensor
from ..general import PosEngFrc


class LogSmapler(nn.Module):
    """Simple line search sampler.
    if stp is too big, stp -> stp * mag
    if stp is too small, stp -> stp / mag
    """
    def __init__(self, mag: float, a0=1.0):
        super().__init__()
        assert 0 < mag < 1
        self.mag = mag
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
        self.stp[flt & (con == 1)] *= self.mag
        self.stp[flt & (con == -1)] /= self.mag
        return self.stp
