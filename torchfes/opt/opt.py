from typing import Dict
import torch
from torch import Tensor, nn
from .. import properties as p
from .eigenvectorfollowing import eigenvector_following
from .derivative import hessian, grad
from .utils import generalize_pos_lag, cartesian_pos_frc_lag
from ..utils import detach
from .quasinewton import bfgs_hes


def set_constraints(inp: Dict[str, Tensor]):
    out = inp.copy()
    if p.con_lag not in inp:
        out[p.con_lag] = torch.zeros_like(out[p.con_cen])
    return out


class Optimizer(nn.Module):
    def __init__(self, vec):
        super().__init__()
        self.vec = vec

    def forward(self, inp: Dict[str, Tensor]):
        out = set_constraints(inp)
        out = generalize_pos_lag(out)
        out = self.vec(out)
        out[p.gen_pos] = out[p.gen_pos] + out[p.gen_stp]
        out = cartesian_pos_frc_lag(out)
        return detach(out)


class EigenVectorFollowing(nn.Module):
    def __init__(self, hes, ord_sdp: int):
        super().__init__()
        self.hes = hes
        self.ord_sdp = ord_sdp  # order of saddle point

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        n_con = inp[p.con_cen].size(1)
        out = self.hes(inp)
        out[p.gen_stp] = eigenvector_following(out[p.gen_hes].detach(),
                                               out[p.gen_grd].detach(),
                                               self.ord_sdp + n_con)
        return out


class NewtonSolve(nn.Module):
    def __init__(self, hes):
        super().__init__()
        self.hes = hes

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        out = self.hes(inp)
        m = torch.tensor(-1.0)
        out[p.gen_stp] = torch.solve(
            out[p.gen_grd][:, :, None], out[p.gen_hes])[0].squeeze(2) * m
        return out


class ExactHessian(nn.Module):
    def __init__(self, lag):
        super().__init__()
        self.lag = lag

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        L = self.lag(out)
        x = inp[p.gen_pos]
        g = grad(L, x, create_graph=True)
        h = hessian(L, x, g)
        out[p.gen_grd] = g
        out[p.gen_hes] = h
        return out


class BFGS(nn.Module):
    def __init__(self, lag, stp: float):
        super().__init__()
        self.lag = lag
        self.stp = stp

    def forward(self, mol: Dict[str, Tensor]):
        L = self.lag(mol)
        x = mol[p.gen_pos]
        g = grad(L, x)
        if p.gen_stp in mol:
            s = mol[p.gen_stp]
            y = g - mol[p.gen_grd]
            h = mol[p.gen_hes]
            h = bfgs_hes(h, s, y)
        else:
            bch, dim = mol[p.gen_pos].size()
            h = torch.eye(dim)[None, :, :].expand([bch, dim, dim]) / self.stp
        mol[p.gen_grd] = g
        mol[p.gen_hes] = h
        return mol


class ConstraintLagrangian(nn.Module):
    def __init__(self, adj, eng, con):
        super().__init__()
        self.adj = adj
        self.eng = eng
        self.con = con

    def forward(self, inp: Dict[str, Tensor]):
        out = self.adj(inp)
        out = self.eng(out)
        con = self.con(out) - out[p.con_cen]
        L = out[p.eng] - (out[p.con_lag] * con).sum(1)
        if p.con_aug in out:
            L = L + (con.pow(2).sum(1) * 0.5 / out[p.con_aug])
        return L
