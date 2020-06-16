"""Not Tested."""
from typing import List, Dict

import torch
from torch import Tensor, nn

from ...general import PosEngFrc, PosEngFrcStorage


class LBFGS(nn.Module):
    def __init__(self, evl, n: int, a: float = 1.0):
        super().__init__()
        self.evl = evl
        self.old = PosEngFrcStorage()
        self.s: List[Tensor] = []
        self.y: List[Tensor] = []
        self.r: List[Tensor] = []
        self.a = a
        self.vec = torch.tensor([])
        self.n = n

    def _init(self, pef: PosEngFrc):
        self.s.clear()
        self.y.clear()
        self.r.clear()
        self.old(pef)
        self.vec = pef.frc * self.a
        return pef, self.vec

    def init(self, pos: Tensor, env: Dict[str, Tensor]):
        _, pef = self.evl(env, pos)
        return self._init(pef)

    def peek(self):
        return self.vec

    def extend(self, pef: PosEngFrc):
        old: PosEngFrc = self.old()
        s = pef.pos - old.pos
        y = old.frc - pef.frc
        r = 1 / (s * y).sum(-1)[:, None]
        self.s.append(s)
        self.y.append(y)
        self.r.append(r)
        self.old(pef)
        if len(self.s) > self.n:
            self.s = self.s[-self.n:]
            self.y = self.y[-self.n:]
            self.r = self.r[-self.n:]
            assert len(self.s) == self.n

    def forward(self, pef: PosEngFrc, env: Dict[str, Tensor], flt: Tensor,
                reset: bool = False):
        assert len(env) > 0
        if reset:
            self._init(pef)
            return self.vec
        if flt.all().item():
            self.extend(pef)
            q = -pef.frc
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
            self.vec = -q
            return self.vec
        else:
            if flt.any().item():
                raise NotImplementedError(
                    'LBFGS for sync=False is not implemented.'
                )
            return self.vec
