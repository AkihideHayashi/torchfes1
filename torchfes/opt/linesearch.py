from torchfes.opt.utils.generalize import generalize
from typing import Dict
import torch
from torch import Tensor, nn
from .. import properties as p
from .utils import (
    Lagrangian, set_directional_hessian, set_directional_gradient)
from ..utils import detach

# Line search Newtons method is below
# gp := g(x + a * p) @ p
# hp := p @ H @ p
#
# while True:
#     eval f(x), g(x), H(x)
#     p = - H^-1 g
#     a = 1
#     tol = gp(0) * tol_dec_rate
#     while |gp(a)| > tol:
#         a = a - gp(a) / hp(a)
#     dx = a * p
#     x = x + dx
#
# while True:
#     eval f(x), g(x), H(x)
#     p = - H^-1 g
#     tol = gp(x) * tol_dec_rate
#     x = x + p
#     eval gp(x)
#     while |gp(x)| > tol:
#         da = - gp(x) / hp(x)
#         x = x + p * da
#
# while True:
#     eval gp(x)
#     if init or |gp(x)| < tol:
#         p = - H^-1 g
#         tol = gp(x) * tol_dec_rate
#         x = x + p
#     else:
#         da = - gp(x) / hp(x)
#         x = x + p * da
#
# while True:
#     x = x + dx
#     eval f(x), g(x)
#     eval gp(x)
#     if init or |gp(x)| < tol:
#         eval H(x)
#         p = - H^-1 g
#         tol = gp(x) * tol_dec_rate
#         dx = p
#     else:
#         eval hp(x)
#         da = - gp(x) / hp(x)
#         dx = p * da


class LineSearch(nn.Module):
    def __init__(self,
                 lag: Lagrangian, dir: nn.Module,
                 tol: float,
                 grd_dec_rat: float = 0.1,
                 lin_ini: bool = True,
                 abs_grd_hes: bool = True,
                 ):
        super().__init__()
        self.lag = lag
        self.grd_dec_rat = grd_dec_rat
        self.dir = dir
        self.first_step_line_search = lin_ini
        self.abs_grd_hes = abs_grd_hes
        self.tol = tol

    def _direction(self, mol: Dict[str, Tensor]):
        mol = self.dir(mol)
        gsd = mol[p.gen_stp_dir]
        n_bch = gsd.size(0)
        mol[p.gen_stp_siz] = torch.ones(
            [n_bch], dtype=gsd.dtype, device=gsd.device)
        mol[p.gen_stp] = mol[p.gen_stp_dir]
        return mol

    def _linesearch(self, mol: Dict[str, Tensor]):
        mol = mol.copy()
        mol = set_directional_hessian(mol)
        if self.abs_grd_hes:
            mol[p.gen_stp_siz] = -mol[p.gen_dir_grd] / mol[p.gen_dir_hes].abs()
        else:
            mol[p.gen_stp_siz] = -mol[p.gen_dir_grd] / mol[p.gen_dir_hes]
        mol[p.gen_stp] = mol[p.gen_stp_siz][:, None] * mol[p.gen_stp_dir]
        return mol

    def _condition(self, mol: Dict[str, Tensor]):
        if p.gen_lin_tol not in mol:
            return True
        if (mol[p.gen_dir_grd].abs() < mol[p.gen_lin_tol]).all():
            return True
        else:
            return False

    def forward(self, mol: Dict[str, Tensor]):
        if p.gen_pos not in mol:
            mol = generalize(mol)
        if p.gen_stp not in mol:
            mol = self.lag(mol, create_graph=True)
            mol = self._direction(mol)
            mol = set_directional_gradient(mol)
            mol[p.gen_lin_tol] = mol[p.gen_dir_grd].abs() * self.grd_dec_rat
            mol[p.gen_lin_tol].masked_fill_(
                mol[p.gen_lin_tol] < self.tol, self.tol)
            if self.first_step_line_search:
                mol = self._linesearch(mol)
        mol[p.gen_pos] = mol[p.gen_pos] + mol[p.gen_stp]
        mol = self.lag(mol, create_graph=True)
        mol = set_directional_gradient(mol)
        if self._condition(mol):
            mol = self._direction(mol)
            mol[p.gen_lin_tol] = mol[p.gen_dir_grd].abs() * self.grd_dec_rat
            mol[p.gen_lin_tol].masked_fill_(
                mol[p.gen_lin_tol] < self.tol, self.tol)
            if self.first_step_line_search:
                mol = self._linesearch(mol)
        else:
            mol = self._linesearch(mol)
        return detach(mol)


def limit_step_size(mol: Dict[str, Tensor], siz: float):
    mol = mol.copy()
    real_dir_siz: Tensor = mol[p.gen_stp_dir].norm(p=2, dim=1)
    idel_dir_siz = real_dir_siz.masked_fill(real_dir_siz > siz, siz)
    ratio = idel_dir_siz / real_dir_siz
    mol[p.gen_stp_siz] = ratio
    mol[p.gen_stp] = mol[p.gen_stp_dir] * mol[p.gen_stp_siz][:, None]
    return mol


class LimitStepSize(nn.Module):
    def __init__(self, dir, siz):
        super().__init__()
        self.siz = siz
        self.dir = dir

    def forward(self, mol: Dict[str, Tensor]):
        siz = self.siz
        mol = self.dir(mol)
        real_dir_siz: Tensor = mol[p.gen_stp_dir].norm(p=2, dim=1)
        idel_dir_siz = real_dir_siz.masked_fill(real_dir_siz > siz, siz)
        ratio = idel_dir_siz / real_dir_siz
        mol[p.gen_stp_dir] = mol[p.gen_stp_dir] * ratio[:, None]
        return mol
