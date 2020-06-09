from typing import Dict, Optional, Tuple
import torch
from torch import nn, Tensor
from .transform import Decoder, Encoder, PosEngFrc, PosEngFrcStorage
from .linesearch import WolfeCondition, LogSmapler
from .solver import (init_conjugate_gradient_step, conjugate_gradient_step,
                     ConjugateGradientStep)
from ..forcefield import EvalEnergiesForces
from .. import properties as p


def optional(tensor: Tensor) -> Optional[Tensor]:
    return tensor


class LineSearchNewtonCG(nn.Module):
    def __init__(self, evl: EvalEnergiesForces, use_cel: bool = False,
                 condition=None, sampler=None, eps: float = 1e-30):
        super().__init__()
        self.evl = evl
        self.encoder = Encoder(use_cel)
        self.decoder = Decoder(use_cel)
        self.eps_min = eps
        self.eps = torch.tensor([])
        self.vec = torch.tensor([])
        self.stp = torch.tensor([])
        self.line_searching = False
        if condition is None:
            condition = WolfeCondition(0.4, 0.6)
        if sampler is None:
            sampler = LogSmapler(1.0, 0.9)
        self.cond = condition
        self.samp = sampler
        self.pef = PosEngFrcStorage()

    def calc(self, inp: Dict[str, Tensor], x: Tensor) -> Tuple[Tensor, Tensor]:
        code: PosEngFrc = self.encoder(inp)
        pos = code.pos.clone().detach().requires_grad_(True)
        out: Dict[str, Tensor] = self.decoder(inp, pos)
        out = self.evl(out, retain_graph=True, create_graph=True, detach=False)
        eng = out[p.eng_tot]
        jac, = torch.autograd.grad(
            [eng.sum()], [pos], retain_graph=True, create_graph=True)
        if jac is None:
            raise RuntimeError()
        else:
            frc = -jac.detach()
            Ax, = torch.autograd.grad([jac], [pos], [optional(x)])
            if Ax is None:
                raise RuntimeError()
            else:
                return frc, Ax

    def init(self, inp: Dict[str, Tensor], vec_ini: Tensor):
        frc, Ax = self.calc(inp, vec_ini)
        return init_conjugate_gradient_step(
            vec_ini[:, :, None], Ax[:, :, None], frc[:, :, None], self.eps)

    def step(self, inp: Dict[str, Tensor], stp: ConjugateGradientStep):
        _, Ap = self.calc(inp, stp.p.squeeze(2))
        return conjugate_gradient_step(stp, Ap[:, :, None], self.eps)

    def forward(self, inp: Dict[str, Tensor]):
        pef_new = self.encoder(inp)
        pef_old = self.pef()
        if pef_old.pos.size() != pef_new.pos.size():
            pef_old = self.pef(pef_new)
            self.vec = pef_old.frc
            self.stp = torch.ones_like(pef_old.eng, dtype=torch.long)
            self.cond.init(pef_old)
            self.newton_cg(inp)
        else:
            cond = self.cond(pef_new, self.stp, self.vec)
            if (cond == 0).all():
                self.newton_cg(inp)
                self.pef(pef_new)
            else:
                self.line_search(cond)
        pos_new = self.pef().pos + self.stp[:, None] * self.vec
        return self.evl(self.decoder(inp, pos_new))

    def line_search(self, cond: Tensor):
        self.stp = self.samp(cond)

    def newton_cg(self, inp: Dict[str, Tensor]):
        pos_eng_frc: PosEngFrc = self.encoder(inp)
        frc = pos_eng_frc.frc
        frc_pow = (frc * frc).sum(-1)
        self.eps = torch.clamp_min(
            torch.clamp_max(frc_pow.sqrt(), 0.25) * frc_pow, self.eps_min)
        if self.vec.size() != pos_eng_frc.pos.size():
            self.vec = pos_eng_frc.frc.detach()
            self.stp = torch.ones_like(pos_eng_frc.eng)
        stp: ConjugateGradientStep = self.init(inp, self.vec)
        while ((stp.r * stp.r).squeeze(2).sum(1) > self.eps).any():
            stp = self.step(inp, stp)
        self.vec = stp.x.squeeze(2)


class NewtonCG(nn.Module):
    def __init__(self, evl: EvalEnergiesForces, use_cel: bool = False):
        super().__init__()
        self.evl = evl
        self.encoder = Encoder(use_cel)
        self.decoder = Decoder(use_cel)
        self.eps = torch.tensor([])
        self.vec = torch.tensor([])

    def calc(self, inp: Dict[str, Tensor], x: Tensor):
        code: PosEngFrc = self.encoder(inp)
        pos = code.pos.clone().detach().requires_grad_(True)
        out: Dict[str, Tensor] = self.decoder(inp, pos)
        out = self.evl(out, retain_graph=True, create_graph=True, detach=False)
        eng = out[p.eng_tot]
        jac, = torch.autograd.grad(
            [eng.sum()], [pos], retain_graph=True, create_graph=True)
        if jac is None:
            raise RuntimeError()
        else:
            frc = -jac.detach()
            Ax, = torch.autograd.grad([jac], [pos], [optional(x)])
            if Ax is None:
                raise RuntimeError()
            else:
                return frc, Ax

    def init(self, inp: Dict[str, Tensor], vec_ini: Tensor):
        frc, Ax = self.calc(inp, vec_ini)
        return init_conjugate_gradient_step(
            vec_ini[:, :, None], Ax[:, :, None], frc[:, :, None], self.eps)

    def step(self, inp: Dict[str, Tensor], stp: ConjugateGradientStep):
        _, Ap = self.calc(inp, stp.p.squeeze(2))
        return conjugate_gradient_step(stp, Ap[:, :, None], self.eps)

    def forward(self, inp: Dict[str, Tensor]):
        pos_eng_frc: PosEngFrc = self.encoder(inp)
        frc = pos_eng_frc.frc
        frc_pow = (frc * frc).sum(-1)
        self.eps = torch.clamp_min(frc_pow.sqrt(), 0.25) * frc_pow
        if self.vec.size() != pos_eng_frc.pos.size():
            self.vec = pos_eng_frc.frc.detach()
        stp: ConjugateGradientStep = self.init(inp, self.vec)
        while ((stp.r * stp.r).unsqueeze(2).sum(1) > self.eps).any():
            stp = self.step(inp, stp)
        self.vec = stp.x.squeeze(2)
        return self.evl(self.decoder(inp, pos_eng_frc.pos + self.vec))
