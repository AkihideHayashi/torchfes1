from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p
from ..utils import grad, requires_grad


def jacobian(fun: Tensor, pos: Tensor, create_graph: bool):
    n = fun.size(1)
    jac = []
    for i in range(n):
        g = torch.zeros_like(fun)
        g[:, i] = 1.0
        grd = grad(fun, pos, g, create_graph=create_graph, retain_graph=True)
        jac.append(grd)
    return torch.stack(jac, 1)


def bme_mmt(mas: Tensor, jac: Tensor):
    """Calculare mass metrics tensor Z."""
    z = (jac[:, :, None, :, :] *
         jac[:, None, :, :, :] /
         mas[:, None, None, :, None]).sum(-1).sum(-1)
    return z


def bme_ktg(pos: Tensor, mas: Tensor, kbt: Tensor,
            jac_con_pos: Tensor, mmt: Tensor, mmt_det: Tensor):
    jac_mmt_det = grad(mmt_det, pos)
    term = (jac_con_pos[:, :, :, :] *
            jac_mmt_det[:, None, :, :] /
            mas[:, None, :, None]).sum(-1).sum(-1)
    term = torch.solve(term[:, :, None], mmt)[0].squeeze(-1)
    g = term / (2 * mmt_det[:, None])
    return kbt[:, None] * g


def fixman(mmt_det: Tensor):
    return mmt_det.pow(-0.5)[:, None]


def bme_kinetic_correction(lmd: Tensor, ktg: Tensor):
    return lmd + ktg


def bme_postprocess(lmd: Tensor, ktg: Tensor, fix: Tensor):
    # n_trj, n_bch, n_col = lmd.size()
    term = lmd + ktg
    return (term * fix).mean(0) / fix.mean(0)


class BMEJac(nn.Module):
    def __init__(self, create_graph: bool):
        super().__init__()
        self.create_graph = create_graph

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        con = out[p.col_var] - out[p.col_cen]
        if p.bme_lmd_tmp not in out:
            out[p.bme_lmd_tmp] = torch.zeros_like(con)
        jac = jacobian(con, out[p.pos], self.create_graph)
        out[p.col_jac] = jac
        return out


class BMEKTGFix(nn.Module):

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        mmt = bme_mmt(out[p.mas], out[p.col_jac])
        mmt_det = mmt.det()
        fix = fixman(mmt_det)
        ktg = bme_ktg(out[p.pos], out[p.mas], out[p.kbt],
                      out[p.col_jac], mmt, mmt_det)
        out = inp.copy()
        out[p.bme_ktg_tmp] = ktg.detach()
        out[p.bme_fix_tmp] = fix.detach()
        return out


class BMEShk(nn.Module):
    def __init__(self, col_var: nn.Module, tol: float, debug=None):
        super().__init__()
        self.col_var = col_var
        self.tol = tol
        self.debug = debug

    def forward(self, inp: Dict[str, Tensor]):
        jac_con = inp[p.col_jac]
        pos = inp[p.pos].clone()
        mom = inp[p.mom].clone()
        out = inp.copy()
        mas = out[p.mas][:, :, None]
        dtm = out[p.dtm][:, None, None]
        lmd = jac_con.new_zeros(jac_con.size()[:2])
        i = 0
        while True:
            lmd.requires_grad_(True)
            frc = -(lmd[:, :, None, None] * jac_con).sum(1)
            out[p.pos] = pos + frc / mas * dtm * dtm
            out[p.mom] = mom + frc * dtm
            out = self.col_var(out)
            con = out[p.col_var] - out[p.col_cen]
            if con.abs().max() < self.tol:
                break
            jac_con_lmd = jacobian(con, lmd, False)
            dlt, _ = torch.solve(con[:, :, None], jac_con_lmd)
            lmd = (lmd.detach() - dlt.squeeze(2).detach())
            if self.debug is not None:
                print(f'Shake step {i}', file=self.debug)
            i += 1
        out[p.pos].detach_()
        out[p.mom].detach_()
        if self.debug is not None:
            print(f'Shake converged after {i} steps.')
        out[p.bme_lmd_tmp] += lmd
        return out


def rattle_objective(mom: Tensor, mas: Tensor, dtm: Tensor, jac: Tensor,
                     lmd: Tensor):
    tmp = mom - (jac * lmd[:, :, None, None]).sum(1) * dtm
    ret = (jac / mas[:, None, :, :] * tmp[:, None, :, :]).sum(-1).sum(-1)
    return ret


class BMERtl(nn.Module):
    def __init__(self, con: nn.Module, tol: float, debug=None):
        super().__init__()
        self.con = con
        self.tol = tol
        self.debug = debug

    def forward(self, inp: Dict[str, Tensor]):
        mom = inp[p.mom].clone()
        out = inp.copy()
        mas = out[p.mas][:, :, None]
        dtm = out[p.dtm][:, None, None]
        jac_con_pos = inp[p.col_jac]
        lmd = jac_con_pos.new_zeros(jac_con_pos.size()[:2])
        i = 0
        while True:
            lmd.requires_grad_(True)
            frc = -(lmd[:, :, None, None] * jac_con_pos).sum(1)
            out[p.mom] = mom + frc * dtm
            dot = rattle_objective(mom, mas, dtm, jac_con_pos, lmd)
            if dot.abs().max() < self.tol:
                break
            jac_dot_lmd = jacobian(dot, lmd, False)
            dlt, _ = torch.solve(dot[:, :, None], jac_dot_lmd)
            lmd = (lmd.detach() - dlt.squeeze(2).detach())
            if self.debug is not None:
                print(f'Rattle step {i}', file=self.debug)
            i += 1
        out[p.mom].detach_()
        if self.debug is not None:
            print(f'Rattle converged after {i} steps.', file=self.debug)
        out[p.bme_lmd_tmp] += lmd
        return out


def bme_det_lmd(inp: Dict[str, Tensor]):
    """Determine blue moon lambda."""
    out = inp.copy()
    if p.bme_lmd_tmp in out:
        out[p.col_mul] = out[p.bme_lmd_tmp]
        out[p.bme_lmd_tmp] = torch.zeros_like(out[p.bme_lmd_tmp])
    if p.bme_fix_tmp in out:
        out[p.bme_fix] = out[p.bme_fix_tmp]
    if p.bme_ktg_tmp in out:
        out[p.bme_ktg] = out[p.bme_ktg_tmp]
    return out
