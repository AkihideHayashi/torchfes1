from typing import Dict

import torch
from torch import Tensor, nn

from .. import properties as p
from ..fes.bme import BMEV, BMEVariables, Rattle, Shake
from .fire import FIRE
from .unified import update_mom, update_pos, update_tim
from ..utils import pnt_ful
from ..forcefield import EvalEnergiesForces


class PQP(nn.Module):
    """NVE Velocity Verlet"""

    def __init__(self, eng: nn.Module, adj: nn.Module):
        super().__init__()
        self.evl = EvalEnergiesForces(eng)
        self.adj = adj

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 0.5)
        out = update_pos(out, 1.0)
        out = update_tim(out, 1.0)
        out = self.evl(out, self.adj(pnt_ful(out)))
        out = update_mom(out, 0.5)
        return out


class PQ(nn.Module):
    """NVE Leap frog."""

    def __init__(self, eng: nn.Module, adj: nn.Module):
        super().__init__()
        self.evl = EvalEnergiesForces(eng)
        self.adj = adj

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 1.0)
        out = update_pos(out, 1.0)
        out = update_tim(out, 1.0)
        out = self.evl(out, self.adj(pnt_ful(out)))
        return out


class PQF(nn.Module):
    """Leap frog FIRE."""

    def __init__(self, eng: nn.Module, adj: nn.Module,
                 a0: float, n_min: int, f_a: float,
                 f_inc: float, f_dec: float, dtm_max: float):
        super().__init__()
        self.evl = EvalEnergiesForces(eng)
        self.adj = adj
        self.fire = FIRE(a0, n_min, f_a, f_inc, f_dec, dtm_max)

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 1.0)
        out = update_pos(out, 1.0)
        out = update_tim(out, 1.0)
        out = self.evl(out, self.adj(pnt_ful(out)))
        out = self.fire(out)
        return out


class PQPF(nn.Module):
    """NVE Velocity Verlet"""

    def __init__(self, eng: nn.Module, adj: nn.Module,
                 a0: float, n_min: int, f_a: float,
                 f_inc: float, f_dec: float, dtm_max: float):
        super().__init__()
        self.evl = EvalEnergiesForces(eng)
        self.adj = adj
        self.fire = FIRE(a0, n_min, f_a, f_inc, f_dec, dtm_max)

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 0.5)
        out = update_pos(out, 1.0)
        out = update_tim(out, 1.0)
        out = self.evl(out, self.adj(pnt_ful(out)))
        out = update_mom(out, 0.5)
        out = self.fire(out)
        return out


class PQTQ(nn.Module):
    """High precision NVT Leap Frog"""

    def __init__(self, eng: nn.Module, adj: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = EvalEnergiesForces(eng)
        self.adj = adj
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 1.0)
        out = update_pos(out, 0.5)
        out = self.kbt(out, 1.0)
        out = update_pos(out, 0.5)
        out = update_tim(out, 1.0)
        out = self.evl(out, self.adj(pnt_ful(out)))
        return out


class PQTQP(nn.Module):
    """High precision NVT Velocity Verlet."""

    def __init__(self, eng: nn.Module, adj: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = EvalEnergiesForces(eng)
        self.adj = adj
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 0.5)
        out = update_pos(out, 0.5)
        out = self.kbt(out, 1.0)
        out = update_pos(out, 0.5)
        out = update_tim(out, 1.0)
        out = self.evl(out, self.adj(pnt_ful(out)))
        out = update_mom(out, 0.5)
        return out


class TPQPT(nn.Module):
    """NVT Velocity Verlet."""

    def __init__(self, eng: nn.Module, adj: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = EvalEnergiesForces(eng)
        self.adj = adj
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = self.kbt(inp, 0.5)
        out = update_mom(out, 0.5)
        out = update_pos(out, 0.5)
        out = update_tim(out, 1.0)
        out = self.evl(out, self.adj(pnt_ful(out)))
        out = update_mom(out, 0.5)
        out = self.kbt(out, 0.5)
        return out


class PTPQ(nn.Module):
    """NVT Leap Frog."""

    def __init__(self, eng: nn.Module, adj: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = EvalEnergiesForces(eng)
        self.adj = adj
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 0.5)
        out = self.kbt(out, 1.0)
        out = update_mom(out, 0.5)
        out = update_tim(out, 1.0)
        out = update_pos(out, 1.0)
        out = self.evl(out, self.adj(pnt_ful(out)))
        return out


class PTPQS(nn.Module):
    """Constrained NVT Leap Frog."""

    def __init__(self, eng: nn.Module, adj: nn.Module,
                 kbt: nn.Module, con: nn.Module, tol: float):
        super().__init__()
        self.evl = EvalEnergiesForces(eng)
        self.adj = adj
        self.kbt = kbt
        self.shk = Shake(con, tol)
        self.bme = BMEVariables(con)

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        out = update_mom(out, 0.5)
        out = self.kbt(out, 1.0)
        out = update_mom(out, 0.5)
        out = update_tim(out, 1.0)

        bme: BMEV = self.bme(out)
        out = update_pos(out, 1.0)
        out, lmd = self.shk(out, bme.jac)
        out = self.evl(out, self.adj(pnt_ful(out)))

        out[p.bme_lmd] = lmd.detach()
        out[p.bme_cor] = bme.cor.detach()
        out[p.bme_fix] = (torch.ones_like(lmd) /
                          bme.mmt.detach().det().sqrt()[:, None])
        return out


class TPQSPTR(nn.Module):
    """Constrained NVT Velocity Verlet."""

    def __init__(self, eng: nn.Module, adj: nn.Module,
                 kbt: nn.Module, con: nn.Module,
                 tol_pos: float, tol_mom: float,
                 ):
        super().__init__()
        self.evl = EvalEnergiesForces(eng)
        self.adj = adj
        self.kbt = kbt
        self.shk = Shake(con, tol_pos)
        self.rtl = Rattle(con, tol_mom)
        self.bme = BMEVariables(con)
        self.lmd_rtl = torch.tensor([0.0])
        self.str_bme_jac = torch.tensor([])
        self.str_bme_fix = torch.tensor([])
        self.str_bme_mmt = torch.tensor([])
        self.str_bme_cor = torch.tensor([])

    def get_bme(self, inp: Dict[str, Tensor]):
        if self.str_bme_jac.numel() == 0:
            self.set_bme(inp)
        return BMEV(jac=self.str_bme_jac, mmt=self.str_bme_mmt,
                    fix=self.str_bme_fix, cor=self.str_bme_cor)

    def set_bme(self, inp: Dict[str, Tensor]):
        bme: BMEV = self.bme(inp)
        self.str_bme_mmt = bme.mmt
        self.str_bme_jac = bme.jac
        self.str_bme_fix = bme.fix
        self.str_bme_cor = bme.cor

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()

        bme: BMEV = self.get_bme(out)

        out = self.kbt(out, 0.5)
        out = update_mom(out, 0.5)
        out = update_tim(out, 1.0)
        out = update_pos(out, 1.0)
        out, lmd = self.shk(out, bme.jac)
        out = self.evl(out, self.adj(pnt_ful(out)))
        out = update_mom(out, 0.5)
        out = self.kbt(out, 0.5)

        self.set_bme(out)
        tmp = self.get_bme(out)
        out, lmd_rtl = self.rtl(out, tmp.jac)

        out[p.bme_lmd] = lmd.detach() + self.lmd_rtl
        out[p.bme_cor] = bme.cor.detach()
        out[p.bme_fix] = bme.fix.detach()
        self.lmd_rtl = lmd_rtl.detach()
        return out
