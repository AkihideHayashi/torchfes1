from typing import List
from typing import Dict, Optional
import torch
from torch import Tensor
from torch import nn
from ..general import PosEngFrc

# pos[t+1] = pos[t] + stp[t] * vec[t]
#
# Standard algorithms using line search is
# ========================================
# pef = evl(encode(env, pos))
# pefs = [pef]
# vec = frc
# main loop:
#     loop for line search:
#         pos_tmp = pos + stp * vec
#         eng_tmp, frc_tmp = evl(encode(env, pos))
#         cond = wolfe_condition(stp, vec, eng, frc, eng_tmp, frc_tmp):
#         if cond == 0:
#             break
#         else:
#             stp = sampler(cond, stp)
#     pef = pos_tmp, eng_tmp, frc_tmp
#     pefs.append(pef)
#     vec = direction(pefs)
# ========================================
#
# ========================================
# It can be transformed to
# ========================================
# if init:
#     vec = vector(inp)
# else:
#     eval(pos + stp * vec)
#     cond(evl)
#     if searching or not cond:
#         stp *= mag
#         searching = True
#     else:
#         vector(pos + stp * vec)
#         searching = False


class LineSearchOptimizer(nn.Module):
    def __init__(self, evl: nn.Module, direction: nn.Module,
                 use_cel: bool = False,
                 condition: Optional[nn.Module] = None,
                 sampler: Optional[nn.Module] = None):
        super().__init__()
        if condition is None:
            condition = WolfeCondition(0.4, 0.6)
        if sampler is None:
            sampler = LogSmapler(1.0, 0.9)
        self.evl = evl
        self.next_pos = NextPosition(direction, condition, sampler)
        self.encoder = Encoder(use_cel)
        self.decoder = Decoder(use_cel)

    def forward(self, inp: Dict[str, Tensor]):
        pos_eng_frc = self.encoder(inp)
        next_pos = self.next_pos(pos_eng_frc)
        out = self.decoder(inp, next_pos)
        out = self.evl(out)
        return out


class NextPosition(nn.Module):
    def __init__(self, direction, condition, sampler):
        super().__init__()
        self.drct = direction
        self.cond = condition
        self.samp = sampler
        self.vec = torch.tensor([])
        self.stp = torch.tensor([])
        self.pef = PosEngFrcStorage()

    def forward(self, inp: PosEngFrc):
        if self.vec.size() != inp.pos.size():
            self.vec = self.drct(inp)
            self.stp = torch.ones_like(inp.eng, dtype=torch.long)
            self.cond.init(inp)
            self.pef(inp)
        else:
            cond = self.cond(inp, self.stp, self.vec)
            if (cond == 0).all():
                self.vec = self.drct(inp)
                self.pef(inp)
            else:
                self.stp = self.samp(cond)
        pef_old = self.pef()
        return pef_old.pos + self.stp[:, None] * self.vec


def dot(a, b):
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
        big = new.eng > old.eng - self.c1 * stp * dot(old.frc, vec)
        sml = dot(new.frc, vec) > self.c2 * dot(old.frc, vec)
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
            self.stp[init] = self.a0
        return self.stp

    def forward(self, con: Tensor, _: PosEngFrc):
        self.stp[con == 1] *= self.mag
        self.stp[con == -1] /= self.mag
        return self.stp
