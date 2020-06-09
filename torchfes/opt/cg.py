from typing import List, Dict
import torch
from torch import nn, Tensor
from ..general import PosEngFrc, PosEngFrcStorage


class FRCG(nn.Module):
    def forward(self, frc_new: Tensor, frc_old: Tensor, _):
        return (frc_new * frc_new).sum(-1) / (frc_old * frc_old).sum(-1)


class PRCG(nn.Module):
    def forward(self, frc_new: Tensor, frc_old: Tensor, _):
        num = (frc_new * (frc_new - frc_old)).sum(-1)
        den = (frc_old * frc_old).sum(-1)
        return num / den


class PRPCG(nn.Module):
    def forward(self, frc_new: Tensor, frc_old: Tensor, _):
        num = (frc_new * (frc_new - frc_old)).sum(-1)
        den = (frc_old * frc_old).sum(-1)
        return torch.clamp_min(num / den, 0.0)


class HSCG(nn.Module):
    def forward(self, frc_new: Tensor, frc_old: Tensor, vec_old: Tensor):
        num = (frc_new * (frc_new - frc_old)).sum(-1)
        den = ((frc_new - frc_old) * vec_old).sum(-1)
        return num / den


class CG(nn.Module):
    def __init__(self, evl, cg_type: nn.Module):
        super().__init__()
        self.evl = evl
        self.cg_type = cg_type
        self.old = PosEngFrcStorage()
        self.vec = torch.tensor([])

    def init(self, pos: Tensor, env: Dict[str, Tensor]):
        pef = self.evl(env, pos)
        self.old(pef)
        self.vec = pef.frc
        return pef.frc, pef

    def forward(self, pef: PosEngFrc, _: Dict[str, Tensor]):
        new = pef
        old: PosEngFrc = self.old()
        beta = self.cg_type(new.frc, old.frc, self.vec)
        vec = new.frc + beta[:, None] * self.vec
        self.vec = vec
        self.old(new)
        return vec
