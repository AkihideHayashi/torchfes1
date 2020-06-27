from typing import Dict

import torch
from torch import Tensor, nn

from ..fes.bme import BMEJac, BMERtl, BMEShk, bme_det_lmd, BMEKTGFix
from .fire import FIRE
from .unified import updt_mom, updt_pos, updt_tim
from ..forcefield import EvalEnergiesForces, EvalEnergies
from .. import properties as p


class AdjEvl(nn.Module):
    """A series of procedures before and after evaluating energy."""

    def __init__(self, adj, evl):
        super().__init__()
        self.adj = adj
        self.evl = evl

    def forward(self, inp: Dict[str, Tensor]):
        out = updt_tim(inp, 1.0)
        out = self.adj(out)
        out = self.evl(out)
        return out


class BMEAdjEvl(nn.Module):
    """A series of procedures before and after evaluating energy."""

    def __init__(self, adj, evl, col, ktg_fix: bool):
        super().__init__()
        self.adj = adj
        self.evl = evl
        self.ktg_fix = ktg_fix
        self.bme_jac = BMEJac(col, ktg_fix)
        self.bme_ktg_fix = BMEKTGFix(col)

    def forward(self, inp: Dict[str, Tensor]):
        out = bme_det_lmd(inp)
        out = updt_tim(out, 1.0)
        out = self.adj(out)
        out = self.evl(out)
        out = self.bme_jac(out)
        if self.ktg_fix:
            out = self.bme_ktg_fix(out)
        return out


class Reset(nn.Module):
    """Reset variables for 1st step."""

    def __init__(self, adj_evl):
        super().__init__()
        self.adj_evl = adj_evl

    def forward(self, inp: Dict[str, Tensor]):
        if p.sld_rst not in inp:
            inp[p.sld_rst] = inp[p.dtm] == inp[p.dtm]
        if inp[p.sld_rst].any():
            out = self.adj_evl(inp)
            out[p.sld_rst] = torch.zeros_like(out[p.sld_rst])
            return out
        else:
            return inp


class PQ(nn.Module):
    """NVE Leap frog."""

    def __init__(self, eng: EvalEnergies, adj: nn.Module):
        super().__init__()
        self.evl = AdjEvl(adj, EvalEnergiesForces(eng))
        self.reset = Reset(self.evl)

    def forward(self, inp: Dict[str, Tensor]):
        out = self.reset(inp)
        out = updt_mom(out, 1.0)
        out = updt_pos(out, 1.0)
        out = self.evl(out)
        return out


class PQP(nn.Module):
    """NVE Velocity Verlet"""

    def __init__(self, eng: EvalEnergies, adj: nn.Module):
        super().__init__()
        self.evl = AdjEvl(adj, EvalEnergiesForces(eng))
        self.reset = Reset(self.evl)

    def forward(self, inp: Dict[str, Tensor]):
        out = self.reset(inp)
        out = updt_mom(out, 0.5)
        out = updt_pos(out, 1.0)
        out = self.evl(out)
        out = updt_mom(out, 0.5)
        return out


class PQF(nn.Module):
    """Leap frog FIRE."""

    def __init__(self, eng: EvalEnergies, adj: nn.Module, fire: FIRE):
        super().__init__()
        self.evl = AdjEvl(adj, EvalEnergiesForces(eng))
        self.reset = Reset(self.evl)
        self.fir = fire

    def forward(self, inp: Dict[str, Tensor]):
        out = self.reset(inp)
        out = updt_mom(out, 1.0)
        out = updt_pos(out, 1.0)
        out = self.evl(out)
        out = self.fir(out)
        return out


class PQTQ(nn.Module):
    """High precision NVT Leap Frog"""

    def __init__(self, eng: EvalEnergies, adj: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = AdjEvl(adj, EvalEnergiesForces(eng))
        self.reset = Reset(self.evl)
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = self.reset(inp)
        out = updt_mom(out, 1.0)
        out = updt_pos(out, 0.5)
        out = self.kbt(out, 1.0)
        out = updt_pos(out, 0.5)
        out = self.evl(out)
        return out


class PQTQP(nn.Module):
    """High precision NVT Velocity Verlet."""

    def __init__(self, eng: EvalEnergies, adj: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = AdjEvl(adj, EvalEnergiesForces(eng))
        self.reset = Reset(self.evl)
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = self.reset(inp)
        out = updt_mom(out, 0.5)
        out = updt_pos(out, 0.5)
        out = self.kbt(out, 1.0)
        out = updt_pos(out, 0.5)
        out = self.evl(out)
        out = updt_mom(out, 0.5)
        return out


class PTPQ(nn.Module):
    """NVT Leap Frog."""

    def __init__(self, eng: EvalEnergies, adj: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = AdjEvl(adj, EvalEnergiesForces(eng))
        self.reset = Reset(self.evl)
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = self.reset(inp)
        out = updt_mom(out, 0.5)
        out = self.kbt(out, 1.0)
        out = updt_mom(out, 0.5)
        out = updt_pos(out, 1.0)
        out = self.evl(out)
        return out


class TPQPT(nn.Module):
    """NVT Velocity Verlet."""

    def __init__(self, eng: EvalEnergies, adj: nn.Module, kbt: nn.Module):
        super().__init__()
        self.evl = AdjEvl(adj, EvalEnergiesForces(eng))
        self.reset = Reset(self.evl)
        self.kbt = kbt

    def forward(self, inp: Dict[str, Tensor]):
        out = self.reset(inp)
        out = self.kbt(out, 0.5)
        out = updt_mom(out, 0.5)
        out = updt_pos(out, 0.5)
        out = self.evl(out)
        out = updt_mom(out, 0.5)
        out = self.kbt(out, 0.5)
        return out


class PTPQs(nn.Module):
    """Constrained NVT Leap Frog."""

    def __init__(self, eng: EvalEnergies, adj: nn.Module,
                 kbt: nn.Module, col_var: nn.Module, tol_shk: float,
                 ktg_fix: bool):
        super().__init__()
        self.evl = BMEAdjEvl(adj, EvalEnergiesForces(eng), col_var, ktg_fix)
        self.reset = Reset(self.evl)
        self.kbt = kbt
        self.shk = BMEShk(col_var, tol_shk)

    def forward(self, inp: Dict[str, Tensor]):
        out = self.reset(inp)
        out = updt_mom(out, 0.5)
        out = self.kbt(out, 1.0)
        out = updt_mom(out, 0.5)
        out = updt_pos(out, 1.0)
        out = self.shk(out)
        out = self.evl(out)
        return out


class TPQsPTr(nn.Module):
    """Constrained NVT Velocity Verlet."""

    def __init__(self, eng: EvalEnergies, adj: nn.Module,
                 kbt: nn.Module, col_var: nn.Module,
                 tol_shk: float, tol_rtl: float, ktg_fix: bool):
        super().__init__()
        self.evl = BMEAdjEvl(adj, EvalEnergiesForces(eng), col_var, ktg_fix)
        self.reset = Reset(self.evl)
        self.kbt = kbt
        self.shk = BMEShk(col_var, tol_shk)
        self.rtl = BMERtl(col_var, tol_rtl)

    def forward(self, inp: Dict[str, Tensor]):
        out = self.reset(inp)
        out = self.kbt(out, 0.5)
        out = updt_mom(out, 0.5)
        out = updt_pos(out, 1.0)
        out = self.shk(out)
        out = self.evl(out)
        out = updt_mom(out, 0.5)
        out = self.kbt(out, 0.5)
        out = self.rtl(out)
        return out


class PQTQs(nn.Module):
    """Constrained high precision NVT Leap Frog."""

    def __init__(self, eng: EvalEnergies, adj: nn.Module,
                 kbt: nn.Module, col_var: nn.Module,
                 tol_shk: float, ktg_fix: bool):
        super().__init__()
        self.evl = BMEAdjEvl(adj, EvalEnergiesForces(eng), col_var, ktg_fix)
        self.reset = Reset(self.evl)
        self.kbt = kbt
        self.shk = BMEShk(col_var, tol_shk)

    def forward(self, inp: Dict[str, Tensor]):
        out = self.reset(inp)
        out = updt_mom(out, 1.0)
        out = updt_pos(out, 0.5)
        out = self.kbt(out, 1.0)
        out = updt_pos(out, 0.5)
        out = self.shk(out)
        out = self.evl(out)
        return out


class PQTQsPr(nn.Module):
    """Constrained high precision NVT Velocity Verlet."""

    def __init__(self, eng: EvalEnergies, adj: nn.Module,
                 kbt: nn.Module, col_var: nn.Module,
                 tol_shk: float, tol_rtl: float, ktg_fix: bool):
        super().__init__()
        self.evl = BMEAdjEvl(adj, EvalEnergiesForces(eng), col_var, ktg_fix)
        self.reset = Reset(self.evl)
        self.kbt = kbt
        self.shk = BMEShk(col_var, tol_shk)
        self.rtl = BMERtl(col_var, tol_rtl)

    def forward(self, inp: Dict[str, Tensor]):
        out = self.reset(inp)
        out = updt_mom(out, 0.5)
        out = updt_pos(out, 0.5)
        out = self.kbt(out, 1.0)
        out = updt_pos(out, 0.5)
        out = self.shk(out)
        out = self.evl(out)
        out = updt_mom(out, 0.5)
        out = self.rtl(out)
        return out
