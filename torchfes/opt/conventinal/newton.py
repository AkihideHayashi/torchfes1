from typing import Dict
import torch
from torch import nn, Tensor
from ... import properties as p
from ..utils import (Lagrangian, set_hessian,
                     set_directional_gradient, set_directional_hessian, solve)


class Newton(nn.Module):
    def __init__(self, lag: Lagrangian):
        super().__init__()
        self.lag = lag

    def forward(self, mol: Dict[str, Tensor]):
        if p.gen_hes not in mol:
            mol = self.lag(mol, create_graph=True)
            mol = set_hessian(mol)
        s = -solve(mol[p.gen_grd], mol[p.gen_hes])
        mol[p.gen_pos] = mol[p.gen_pos] + s
        mol = self.lag(mol, create_graph=True)
        mol = set_hessian(mol)
        return mol


class LineSearchNewton(nn.Module):
    def __init__(self, lag: Lagrangian, grd_dec_rat: float = 0.1):
        super().__init__()
        self.lag = lag
        self.grd_dec_rat = grd_dec_rat

    def _direction(self, mol: Dict[str, Tensor]):
        mol = mol.copy()
        mol = set_hessian(mol)
        mol[p.gen_stp_dir] = -solve(mol[p.gen_grd], mol[p.gen_hes])
        mol[p.gen_stp_siz] = torch.ones_like(mol[p.gen_stp_dir][:, 0])
        mol[p.gen_stp] = mol[p.gen_stp_siz][:, None] * mol[p.gen_stp_dir]
        return mol

    def _linesearch(self, mol: Dict[str, Tensor]):
        mol = mol.copy()
        mol = set_directional_hessian(mol)
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
        if p.gen_stp not in mol:
            mol = self.lag(mol, create_graph=True)
            mol = self._direction(mol)
            mol = set_directional_gradient(mol)
            mol[p.gen_lin_tol] = mol[p.gen_dir_grd].abs() * self.grd_dec_rat
        mol[p.gen_pos] = mol[p.gen_pos] + mol[p.gen_stp]
        mol = self.lag(mol, create_graph=True)
        mol = set_directional_gradient(mol)
        if self._condition(mol):
            mol = self._direction(mol)
            mol[p.gen_lin_tol] = mol[p.gen_dir_grd].abs() * self.grd_dec_rat
        else:
            mol = self._linesearch(mol)
        return mol
