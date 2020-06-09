from typing import Dict, Optional
import torch
from torch import Tensor
from torch import nn
from .transform import PosEngFrc, PosEngFrcStorage, Encoder, Decoder

# Standard algorithms using line search is
# ========================================
# vec = frc
# loop:
#     loop:
#         eval(pos + stp * vec)
#         cond(evl)
#         if cond:
#             break
#         else:
#             stp *= mag
#     vector(pos + stp * vec)
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


class WolfeCondition(nn.Module):
    def __init__(self, c1: float, c2: float):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        assert 0 < c1 < c2 < 1
        self.old = PosEngFrcStorage()

    def init(self, inp: PosEngFrc):
        self.old(inp)

    def forward(self, new: PosEngFrc, stp: Tensor, vec: Tensor):
        old: PosEngFrc = self.old()
        if old.pos.size() != new.pos.size():
            self.old(new)
            old = self.old()
        big = new.eng > old.eng - self.c1 * stp * (old.frc * vec).sum(-1)
        sml = (new.frc * vec).sum(-1) > self.c2 * (old.frc * vec).sum(-1)
        assert not bool((big & sml).any())
        return (big.to(torch.long) - sml.to(torch.long)).to(new.eng)


class LogSmapler(nn.Module):
    def __init__(self, stp_ini: float, mag: float):
        super().__init__()
        self.stp_ini = stp_ini
        self.stp = torch.tensor([])
        assert 0 < mag < 1
        self.mag = mag

    def forward(self, cond: Tensor):
        if self.stp.size() != cond.size():
            self.stp = torch.ones_like(cond)
        self.stp[cond == 1] *= self.mag
        self.stp[cond == -1] /= self.mag
        return self.stp


# def sumsum(tensor):
#     return tensor.sum(-1).sum(-1)


# def dot(x, y):
#     return (x * y).sum(-1).sum(-1)


# def wolfe_condition(eng_new: Tensor, frc_new: Tensor,
#                     eng_old: Tensor, frc_old: Tensor,
#                     stp: Tensor, vec: Tensor,
#                     c1: float, c2: float):
#     assert 0 < c1 < c2 < 1
#     big = eng_new > eng_old - c1 * stp * dot(frc_old, vec)
#     sml = dot(frc_new, vec) > c2 * dot(frc_old, vec)
#     if (big & sml).any():
#         print(eng_old, eng_new)
#         print(big & sml)
#         raise RuntimeError()
#     return big.to(torch.long) - sml.to(torch.long)


# class WolfeMagLineSearchOld(nn.Module):
#     def __init__(self, evl, rho: float, c1: float, c2: float,
#                  stp_ini: float = 1.0):
#         super().__init__()
#         self.evl = evl
#         self.stp = torch.tensor([])
#         self.pos = torch.tensor([])
#         self.eng = torch.tensor([])
#         self.frc = torch.tensor([])
#         self.vec = torch.tensor([])
#         self.rho = rho
#         self.c1 = c1
#         self.c2 = c2
#         self.searching = False
#         self.stp_ini = stp_ini

#     def forward(self, inp: Dict[str, Tensor], vec: Tensor):
#         self.register(inp, vec)
#         out = self.step(inp)
#         wolfe = self.wolfe(out)
#         if (wolfe == 0).all():
#             self.reset()
#             return out, self.stp, False
#         self.stp[wolfe == 1] *= self.rho
#         self.stp[wolfe == -1] /= self.rho
#         return out, self.stp, True

#     def register(self, inp: Dict[str, Tensor], vec: Tensor):
#         if self.pos.size() == inp[p.pos].size():
#             return
#         self.pos = inp[p.pos]
#         self.vec = vec
#         self.eng = inp[p.eng_tot]
#         self.frc = inp[p.frc]
#         self.searching = True
#         if self.stp.size() != self.pos.size():
#             self.stp = torch.ones_like(self.pos) * self.stp_ini

#     def reset(self):
#         self.pos = torch.tensor([])
#         self.eng = torch.tensor([])
#         self.frc = torch.tensor([])
#         self.vec = torch.tensor([])
#         self.searching = False

#     def wolfe(self, inp: Dict[str, Tensor]):
#         wolfe = wolfe_condition(
#             inp[p.eng_tot], inp[p.frc], self.eng, self.frc,
#             self.stp, self.vec, self.c1, self.c2)
#         return wolfe

#     def step(self, inp: Dict[str, Tensor]):
#         out = inp.copy()
#         out[p.pos] = self.pos + self.stp * self.vec
#         out = self.evl(out)
#         return out
