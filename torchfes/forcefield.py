import warnings
from typing import Dict, List

import torch
from torch import Tensor, nn

from . import properties as p
from .api import Energies
from .general import PosEngFrc
from .utils import detach_, grad


class EvalEnergies(nn.Module):
    def __init__(self, mdl, res: List[nn.Module]):
        super().__init__()
        self.mdl = mdl
        self.res = nn.ModuleList(res)

    def forward(self, inp: Dict[str, Tensor]):
        eng_mdl: Energies = self.mdl(inp)
        eng_tot = eng_mdl.eng_mol
        eng_res = torch.zeros_like(eng_tot)
        for res in self.res:
            eng_res = eng_res + res(inp)
        eng_tot = eng_tot + eng_res
        out = inp.copy()
        out[p.eng_atm] = eng_mdl.eng_atm
        out[p.eng_mol] = eng_mdl.eng_mol
        out[p.eng_atm_std] = eng_mdl.eng_atm_std
        out[p.eng_mol_std] = eng_mdl.eng_mol_std
        out[p.eng_res] = eng_res
        out[p.eng_tot] = eng_tot
        return out


class EvalEnergiesForces(nn.Module):
    def __init__(self, mdl: nn.Module, res: List[nn.Module]):
        super().__init__()
        self.eng = EvalEnergies(mdl, res)

    def forward(self, inp: Dict[str, Tensor], frc_pos: bool = True,
                frc_cel: bool = False, frc_grd: bool = False):
        pos = inp[p.pos]
        cel = inp[p.cel]
        if frc_pos and not pos.requires_grad:
            pos.requires_grad_()
        if frc_cel and not cel.requires_grad:
            cel.requires_grad_()
        out = self.eng(inp)
        eng_tot = out[p.eng_tot]
        one = torch.ones_like(eng_tot)
        if frc_pos:
            out[p.frc] = grad(-eng_tot, pos, one, frc_grd)
        if frc_cel:
            out[p.frc_cel] = grad(-eng_tot, cel, one, frc_grd)
        if not frc_grd:
            detach_(out)
        return out


class EvalEnergiesForcesGeneral(nn.Module):
    def __init__(self, mdl: nn.Module, res: List[nn.Module], gen: nn.Module):
        super().__init__()
        self.eng = EvalEnergies(mdl, res)
        self.gen = gen

    def forward(self, env: Dict[str, Tensor], pos: Tensor,
                frc: bool = True, frc_grd: bool = False) -> PosEngFrc:
        if frc and not pos.requires_grad:
            pos = pos.clone().requires_grad_()
        inp = self.gen(env, pos)
        out = self.eng(inp)
        eng_tot = out[p.eng_tot]
        one = torch.ones_like(eng_tot)
        if frc:
            frc_ = grad(-eng_tot, pos, one, frc_grd)
        else:
            frc_ = torch.zeros_like(pos)
        if not frc_grd:
            detach_(out)
            return PosEngFrc(pos=pos.clone().detach(),
                             eng=eng_tot.clone().detach().unsqueeze(-1),
                             frc=frc_.clone().detach())
        else:
            return PosEngFrc(pos=pos, eng=eng_tot.unsqueeze(-1), frc=frc_)


class EvalForcesOnly(nn.Module):
    def __init__(self, evl):
        super().__init__()
        self.evl = evl
        warnings.warn("EvalForcesOnly is obsolete.")

    def forward(self, inp: Dict[str, Tensor]):
        pos = inp[p.pos].clone().detach().requires_grad_(True)
        out: Dict[str, Tensor] = inp.copy()
        out[p.pos] = pos
        out = self.evl(out)
        eng_tot = out[p.eng_tot]
        frc, = torch.autograd.grad([-eng_tot.sum()], [pos])
        if frc is None:
            raise RuntimeError()
        else:
            out[p.frc] = frc.detach()
            for key in out:
                out[key] = out[key].detach()
            return out
