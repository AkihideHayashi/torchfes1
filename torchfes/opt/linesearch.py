from typing import Dict

import torch
from torch import Tensor, nn

from ..general import PosEngFrc, PosEngFrcStorage, where_pef


class LineSearchOptimizer(nn.Module):
    def __init__(self, evl, vec, stp, con, reset, sync: bool):
        super().__init__()
        self.evl = evl
        self.vec = vec
        self.stp = stp
        self.con = con
        self.pef = PosEngFrcStorage()
        self.reset = reset
        self.n_vec = 0
        self.sync = sync

    def init(self, env: Dict[str, Tensor], pos: Tensor):
        pef, _ = self.vec.init(pos, env)
        self.pef(pef)
        self.stp.init(pef, pef.eng == pef.eng, self.reset)

    def get_flt_vec(self, con: Tensor):
        if self.sync:
            if (con == 0).all():
                return torch.ones_like(con, dtype=torch.bool)
            else:
                return torch.zeros_like(con, dtype=torch.bool)
        else:
            return con == 0

    def forward(self, env: Dict[str, Tensor]):
        pef = self.pef()
        stp = self.stp.peek()
        vec = self.vec.peek()
        pos_tmp = pef.pos + stp * vec
        pef_tmp = self.evl(env, pos_tmp)
        con = self.con(pef, pef_tmp, stp, vec)

        flt_vec = self.get_flt_vec(con)
        flt_stp = ~flt_vec

        pef = pef_tmp
        vec = self.vec(pef, env, flt_vec)
        stp = self.stp.init(pef, flt_vec, self.reset)
        self.n_vec += 1

        stp = self.stp(con, pef_tmp, flt_stp)

        self.pef(where_pef(flt_vec, pef, self.pef()))
        return pef_tmp


def _dot(a, b):
    return (a * b).sum(-1).unsqueeze(-1)


class WolfeCondition(nn.Module):
    def __init__(self, c1: float, c2: float):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        assert 0 < c1 < c2 < 1

    def forward(self, old: PosEngFrc, new: PosEngFrc,
                stp: Tensor, vec: Tensor):
        assert torch.allclose(old.pos + stp * vec, new.pos)
        big = new.eng > old.eng - self.c1 * stp * _dot(old.frc, vec)
        sml = _dot(new.frc, vec) > self.c2 * _dot(old.frc, vec)
        assert not bool((big & sml).any())
        return (big.to(torch.long) - sml.to(torch.long)).to(new.eng)


class LogSmapler(nn.Module):
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
            self.stp[init] = torch.ones_like(
                init, dtype=self.stp.dtype) * self.a0
        return self.stp

    def peek(self):
        return self.stp

    def forward(self, con: Tensor, _: PosEngFrc, flt: Tensor):
        self.stp[flt & (con == 1)] *= self.mag
        self.stp[flt & (con == -1)] /= self.mag
        return self.stp
