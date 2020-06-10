"""Not Tested."""
import warnings
from typing import List

from torch import Tensor, nn

from .transform import PosEngFrc, PosEngFrcStorage

warnings.warn("LBFGS is not pass test.")


class LBFGS(nn.Module):
    def __init__(self, stp: float = 1.0):
        super().__init__()
        self.old = PosEngFrcStorage()
        self.s: List[Tensor] = []
        self.y: List[Tensor] = []
        self.r: List[Tensor] = []
        self.stp = stp

    def init(self, inp: PosEngFrc):
        self.s.clear()
        self.y.clear()
        self.r.clear()
        self.old(inp)

    def is_init(self):
        old: PosEngFrc = self.old()
        ret = old.pos.size(0) <= 0
        return ret

    def extend(self, new: PosEngFrc):
        old: PosEngFrc = self.old()
        s = new.pos - old.pos
        y = old.frc - new.frc
        r = 1 / (s * y).sum(-1)[:, None]
        self.s.append(s)
        self.y.append(y)
        self.r.append(r)
        self.old(new)

    def forward(self, new: PosEngFrc):
        print('lbfgs')
        if self.is_init():
            self.init(new)
            return new.frc * self.stp
        self.extend(new)
        q = -new.frc
        n = len(self.y)
        a = []
        for i in range(n - 1, -1, -1):
            si = self.s[i]
            yi = self.y[i]
            ri = self.r[i]
            ai = ri * (si * q).sum(-1)[:, None]
            q = q - ai * yi
            a.append(ai)
        g = ((self.s[-1] * self.y[-1]).sum(-1) /
             (self.y[-1] * self.y[-1]).sum(-1))[:, None]
        q = g * q
        a.reverse()
        for i in range(n):
            si = self.s[i]
            yi = self.y[i]
            ri = self.r[i]
            bi = ri * (yi * q).sum(-1)[:, None]
            q = q + si * (a[i] - bi)
        return -q
