from typing import Dict
import torch
from torch import nn, Tensor
from .unified import update_mom, update_pos, update_tim
from .bme import Shake, BMEV, BMEVariables, Rattle
from . import properties as p
from .forcefield import EvalForcesOnly


class PQP(nn.Module):
    """NVE Velocity Verlet"""

    def __init__(self, evl: nn.Module):
        super().__init__()
        self.evl = evl

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 0.5)
        out = update_pos(out, 1.0)
        out = update_tim(out, 1.0)
        out = self.evl(out)
        out = update_mom(out, 0.5)
        return out


class PQ(nn.Module):
    """NVE Leap frog."""

    def __init__(self, evl: nn.Module):
        super().__init__()
        self.evl = evl

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 1.0)
        out = update_pos(out, 1.0)
        out = update_tim(out, 1.0)
        out = self.evl(out)
        return out


class PQTQ(nn.Module):
    """High precision NVT Leap Frog"""

    def __init__(self, evl: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = evl
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 1.0)
        out = update_pos(out, 0.5)
        out = self.kbt(out, 1.0)
        out = update_pos(out, 0.5)
        out = update_tim(out, 1.0)
        out = self.evl(out)
        return out


class PQTQP(nn.Module):
    """High precision NVT Velocity Verlet."""

    def __init__(self, evl: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = evl
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 0.5)
        out = update_pos(out, 0.5)
        out = self.kbt(out, 1.0)
        out = update_pos(out, 0.5)
        out = update_tim(out, 1.0)
        out = self.evl(out)
        out = update_mom(out, 0.5)
        return out


class TPQPT(nn.Module):
    """NVT Velocity Verlet."""
    def __init__(self, evl: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = evl
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = self.kbt(inp, 0.5)
        out = update_mom(out, 0.5)
        out = update_pos(out, 0.5)
        out = update_tim(out, 1.0)
        out = self.evl(out)
        out = update_mom(out, 0.5)
        out = self.kbt(out, 0.5)
        return out


class PTPQ(nn.Module):
    """NVT Leap Frog."""
    def __init__(self, evl: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = evl
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = update_mom(inp, 0.5)
        out = self.kbt(out, 1.0)
        out = update_mom(out, 0.5)
        out = update_tim(out, 1.0)
        out = update_pos(out, 1.0)
        out = self.evl(out)
        return out


class PTPQS(nn.Module):
    """Constrained NVT Leap Frog."""

    def __init__(self, evl: nn.Module, kbt: nn.Module, con: nn.Module,
                 tol: float):
        super().__init__()
        self.evl = evl
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
        out = self.evl(out)

        out[p.bme_lmd] = lmd.detach()
        out[p.bme_cor] = bme.cor.detach()
        out[p.bme_fix] = (torch.ones_like(lmd) /
                          bme.mmt.detach().det().sqrt()[:, None])
        return out


class TPQSPTR(nn.Module):
    """Constrained NVT Velocity Verlet."""

    def __init__(self, evl: nn.Module, kbt: nn.Module, con: nn.Module,
                 tol_pos: float, tol_mom: float,
                 ):
        super().__init__()
        self.evl = evl
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
        out = self.evl(out)
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