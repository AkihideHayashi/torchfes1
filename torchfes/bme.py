import logging
from itertools import count
from typing import Dict, List, NamedTuple
from torch import nn, Tensor
import torch
from . import properties as p


_logger = logging.getLogger(__name__)


class BMEV(NamedTuple):
    jac: Tensor  # pd sigma / pd r
    mmt: Tensor  # mass-metric tensor "z".
    cor: Tensor  # g
    fix: Tensor  # fixman correction z^(-0.5)


def fixman(mmt: Tensor):
    return mmt.detach().det().pow(-0.5)[:, None]


def jacobian(fun, pos):
    n = fun.size(1)
    jac = []
    for i in range(n):
        g = torch.zeros_like(fun)
        g[:, i] = 1.0
        grad, = torch.autograd.grad(fun, pos, g,
                                    retain_graph=True, create_graph=True)
        jac.append(grad)
    return torch.stack(jac, 1)


def requires_grad(inp: Dict[str, Tensor], props: List[str]):
    out = inp.copy()
    for prop in props:
        out[prop] = inp[prop].clone().detach().requires_grad_(True)
    return out


def bme_z(mas: Tensor, jac: Tensor):
    z = (jac[:, :, None, :, :] *
         jac[:, None, :, :, :] /
         mas[:, None, None, :, None]).sum(-1).sum(-1)
    return z


def bme_g(pos: Tensor, mas: Tensor, jac: Tensor, z: Tensor):
    zd = z.det()
    rz, = torch.autograd.grad(zd, pos, torch.ones_like(zd))
    term = (jac[:, :, :, :] *
            rz[:, None, :, :] /
            mas[:, None, :, None]).sum(-1).sum(-1)
    term = torch.solve(term[:, :, None], z)[0].squeeze(-1)
    term = term / (2 * zd[:, None])
    return term


class BMEVariables(nn.Module):
    def __init__(self, con: nn.Module):
        super().__init__()
        self.con = con

    def forward(self, inp: Dict[str, Tensor]):
        out = requires_grad(inp, [p.pos])
        con = self.con(out)
        jac = jacobian(con, out[p.pos])
        mmt = bme_z(out[p.mas], jac)
        cor = bme_g(out[p.pos], out[p.mas], jac, mmt) * inp[p.kbt][:, None]
        fix = fixman(mmt)
        return BMEV(jac.detach(), mmt.detach(),
                    cor.detach(), fix.detach())


def rattle_objective(mom: Tensor, mas: Tensor, dtm: Tensor,
                     jac_prv: Tensor, lmd: Tensor):
    tmp = mom - (jac_prv * lmd[:, :, None, None]).sum(1) * dtm
    ret = (jac_prv / mas[:, None, :, :] * tmp[:, None, :, :]).sum(-1).sum(-1)
    return ret


class Rattle(nn.Module):
    def __init__(self, con: nn.Module, tol: float):
        super().__init__()
        self.con = con
        self.tol = tol

    def forward(self, inp: Dict[str, Tensor], jac_prv: Tensor):
        mom = inp[p.mom].clone()
        out = inp.copy()
        mas = out[p.mas][:, :, None]
        dtm = out[p.dtm][:, None, None]
        lmd = jac_prv.new_zeros(jac_prv.size()[:2])
        i = 0
        for i in count():
            lmd.requires_grad_(True)
            frc = -(lmd[:, :, None, None] * jac_prv).sum(1)
            out[p.mom] = mom + frc * dtm
            con = rattle_objective(mom, mas, dtm, jac_prv, lmd)
            if con.abs().max() < self.tol:
                break
            jac = jacobian(con, lmd)
            dlt, _ = torch.solve(con[:, :, None], jac)
            lmd = (lmd.detach() - dlt.squeeze(2).detach())
            _logger.debug('Rattle step %d.', i)
        out[p.mom].detach_()
        _logger.debug('Rattle converged after %d steps.', i)
        return out, lmd


class Shake(nn.Module):
    def __init__(self, con: nn.Module, tol: float):
        super().__init__()
        self.con = con
        self.tol = tol

    def forward(self, inp: Dict[str, Tensor], jac_prv: Tensor):
        pos = inp[p.pos].clone()
        mom = inp[p.mom].clone()
        out = inp.copy()
        mas = out[p.mas][:, :, None]
        dtm = out[p.dtm][:, None, None]
        lmd = jac_prv.new_zeros(jac_prv.size()[:2])
        i = 0
        for i in count():
            lmd.requires_grad_(True)
            frc = -(lmd[:, :, None, None] * jac_prv).sum(1)
            out[p.pos] = pos + frc / mas * dtm * dtm
            out[p.mom] = mom + frc * dtm
            con = self.con(out)
            if con.abs().max() < self.tol:
                break
            jac = jacobian(con, lmd)
            dlt, _ = torch.solve(con[:, :, None], jac)
            lmd = (lmd.detach() - dlt.squeeze(2).detach())
            _logger.debug('Shake step %d.', i)
        out[p.pos].detach_()
        out[p.mom].detach_()
        _logger.debug('Shake converged after %d steps.', i)
        return out, lmd
