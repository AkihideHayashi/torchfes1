from typing import Dict
import torch
from torch import nn, Tensor
from .functional import bfgs_inv, lbfgs
from .. import properties as p
from .utils import set_hessian, dot, set_directional_hessian


class ExactBFGS(nn.Module):
    def forward(self, mol: Dict[str, Tensor]):
        mol = mol.copy()
        if p.gen_hes_inv not in mol:
            mol = set_hessian(mol)
            mol[p.gen_hes_inv] = mol.pop(p.gen_hes).inverse()
        else:
            s = mol[p.gen_pos] - mol[p.gen_pos_pre]
            y = mol[p.gen_grd] - mol[p.gen_grd_pre]
            mol[p.gen_hes_inv] = bfgs_inv(mol[p.gen_hes_inv], s, y)
        mol[p.gen_pos_pre] = mol[p.gen_pos]
        mol[p.gen_grd_pre] = mol[p.gen_grd]
        mol[p.gen_stp_dir] = -dot(mol[p.gen_grd], mol[p.gen_hes_inv])
        return mol


class BFGS(nn.Module):
    def __init__(self, ini_stp: float = 0.0):
        super().__init__()
        self.ini_stp = ini_stp

    def forward(self, mol: Dict[str, Tensor]):
        mol = mol.copy()
        if p.gen_hes_inv not in mol:
            bch, dim = mol[p.gen_grd].size()
            eye = torch.eye(dim).to(mol[p.gen_grd])[None].expand(
                [bch, dim, dim])
            if self.ini_stp != 0.0:
                mol[p.gen_hes_inv] = eye * self.ini_stp
            else:
                mol[p.gen_stp_dir] = (-mol[p.gen_grd] /
                                      mol[p.gen_grd].norm(p=2, dim=1)[:, None])
                mol = set_directional_hessian(mol)
                mol[p.gen_hes_inv] = eye / mol[p.gen_dir_hes][:, None, None]
                mol.pop(p.gen_stp_dir)
        else:
            s = mol[p.gen_pos] - mol[p.gen_pos_pre]
            y = mol[p.gen_grd] - mol[p.gen_grd_pre]
            mol[p.gen_hes_inv] = bfgs_inv(mol[p.gen_hes_inv], s, y)
        mol[p.gen_pos_pre] = mol[p.gen_pos]
        mol[p.gen_grd_pre] = mol[p.gen_grd]
        mol[p.gen_stp_dir] = -dot(mol[p.gen_grd], mol[p.gen_hes_inv])
        return mol


class LBFGS(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, mol: Dict[str, Tensor]):
        mol = mol.copy()
        if p.gen_pos_pre not in mol:
            mol[p.gen_pos_pre] = mol[p.gen_pos]
            mol[p.gen_grd_pre] = mol[p.gen_grd]
            mol[p.gen_stp_dir] = -mol[p.gen_grd]
            return mol

        s = mol[p.gen_pos] - mol[p.gen_pos_pre]
        y = mol[p.gen_grd] - mol[p.gen_grd_pre]
        r = (s * y).sum(1, keepdim=True)
        if p.gen_dlt_pos in mol:
            mol[p.gen_dlt_pos] = torch.cat([mol[p.gen_dlt_pos], s[None]], 0)
            mol[p.gen_dlt_grd] = torch.cat([mol[p.gen_dlt_grd], y[None]], 0)
            mol[p.gen_dlt_dot] = torch.cat([mol[p.gen_dlt_dot], r[None]], 0)
        else:
            mol[p.gen_dlt_pos] = s[None]
            mol[p.gen_dlt_grd] = y[None]
            mol[p.gen_dlt_dot] = r[None]
        if mol[p.gen_dlt_pos].size(0) > self.n:
            mol[p.gen_dlt_pos] = mol[p.gen_dlt_pos][-self.n:, :, :]
            mol[p.gen_dlt_grd] = mol[p.gen_dlt_grd][-self.n:, :, :]
            mol[p.gen_dlt_dot] = mol[p.gen_dlt_dot][-self.n:, :, :]
        mol[p.gen_pos_pre] = mol[p.gen_pos]
        mol[p.gen_grd_pre] = mol[p.gen_grd]

        mol[p.gen_stp_dir] = lbfgs(mol[p.gen_grd], mol[p.gen_dlt_pos],
                                   mol[p.gen_dlt_grd], mol[p.gen_dlt_dot])
        return mol


class LBFGSEnsemble(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, mol: Dict[str, Tensor]):
        mol = mol.copy()
        if p.gen_pos_pre not in mol:
            mol[p.gen_pos_pre] = mol[p.gen_pos]
            mol[p.gen_grd_pre] = mol[p.gen_grd]
            mol[p.gen_stp_dir] = -mol[p.gen_grd]
            return mol

        s = mol[p.gen_pos] - mol[p.gen_pos_pre]
        y = mol[p.gen_grd] - mol[p.gen_grd_pre]
        r = (s * y).sum(1, keepdim=True)
        if p.gen_dlt_pos in mol:
            mol[p.gen_dlt_pos] = torch.cat([mol[p.gen_dlt_pos], s[None]], 0)
            mol[p.gen_dlt_grd] = torch.cat([mol[p.gen_dlt_grd], y[None]], 0)
            mol[p.gen_dlt_dot] = torch.cat([mol[p.gen_dlt_dot], r[None]], 0)
        else:
            mol[p.gen_dlt_pos] = s[None]
            mol[p.gen_dlt_grd] = y[None]
            mol[p.gen_dlt_dot] = r[None]
        if mol[p.gen_dlt_pos].size(0) > self.n:
            mol[p.gen_dlt_pos] = mol[p.gen_dlt_pos][-self.n:, :, :]
            mol[p.gen_dlt_grd] = mol[p.gen_dlt_grd][-self.n:, :, :]
            mol[p.gen_dlt_dot] = mol[p.gen_dlt_dot][-self.n:, :, :]
        mol[p.gen_pos_pre] = mol[p.gen_pos]
        mol[p.gen_grd_pre] = mol[p.gen_grd]

        mol[p.gen_stp_dir] = lbfgs(
            _ens(mol[p.gen_grd]), _ens(mol[p.gen_dlt_pos]),
            _ens(mol[p.gen_dlt_grd]), _ens(mol[p.gen_dlt_dot])).view_as(
                mol[p.gen_grd])
        return mol


def _ens(vec: Tensor):
    if vec.dim() == 2:
        bch, dim = vec.size()
        return vec.view([1, bch * dim])
    else:
        assert vec.dim() == 3
        his, bch, dim = vec.size()
        return vec.view([his, 1, bch * dim])
