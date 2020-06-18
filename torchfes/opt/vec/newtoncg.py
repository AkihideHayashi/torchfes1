from typing import Dict

import torch
from torch import Tensor, nn

from ...forcefield import EvalEnergiesForcesGeneral
from ..solver import (conjugate_gradient_step, init_conjugate_gradient_step)
from ...general import PosEngFrc
from ...utils import grad


class NewtonCG(nn.Module):
    def __init__(self, evl: EvalEnergiesForcesGeneral, eps: float = 1e-30):
        super().__init__()
        self.evl = evl
        self.eps = eps
        self.vec = torch.tensor([])

    def get_eps(self, frc: Tensor):
        frc_pow = (frc * frc).sum(-1)
        return torch.clamp_min(
            torch.clamp_max(frc_pow.sqrt(), 0.25) * frc_pow, self.eps)

    def newton_cg(self, pos: Tensor, env: Dict[str, Tensor], vec: Tensor):
        _, pef = self.evl(env, pos, frc=True, frc_grd=True)
        frc = pef.frc.clone().detach()
        eps = self.get_eps(frc)
        Ax = grad(
            -pef.frc, pef.pos, vec, False, retain_graph=True).unsqueeze(-1)
        cg = init_conjugate_gradient_step(
            vec.unsqueeze(-1), Ax, frc.unsqueeze(-1), eps)
        while ((cg.r * cg.r).squeeze(2).sum(1) > eps).any().item():
            Ap = grad(
                -pef.frc, pef.pos, cg.p.squeeze(-1), False, retain_graph=True
            ).unsqueeze(-1)
            cg = conjugate_gradient_step(cg, Ap, eps)
        return cg.x.squeeze(-1)

    def init(self, pos: Tensor, env: Dict[str, Tensor]):
        _, pef = self.evl(env, pos)
        self.vec = pef.frc
        self.vec = self.newton_cg(pos, env, self.vec)
        return pef, self.vec

    def peek(self):
        return self.vec

    def forward(self, pef: PosEngFrc, env: Dict[str, Tensor], flt: Tensor,
                reset: bool = False):
        assert isinstance(reset, bool)
        if not flt.any():
            return self.vec
        vec = self.newton_cg(pef.pos, env, self.vec)
        self.vec = torch.where(flt, vec, self.vec)
        return self.vec
