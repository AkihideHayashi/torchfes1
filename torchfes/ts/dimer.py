from typing import Dict
import math
import torch
from torch import nn, Tensor
from scipy.sparse.linalg import eigs, LinearOperator, eigsh, lobpcg
from ..data import cat, unbind
from .. import properties as p
from ..utils import grad


def normalize(x: Tensor):
    return x / x.flatten(1).norm(p=2, dim=1)[:, None, None]


def orthogonalized_delta_force(nhn: Tensor, hn: Tensor, n: Tensor):
    return hn - nhn[:, None, None] * n


def rotate_angle(nhn: Tensor, ghn: Tensor, ghg: Tensor):
    B: Tensor = 0.5 * (nhn - ghg)
    A = ghn
    AB = (B / (A * A + B * B).sqrt()).asin()
    phi = torch.where(A >= 0, AB, math.pi - AB)
    return near_zero(-math.pi / 4 - phi / 2)


def near_zero(theta):
    return (theta + math.pi / 2) % math.pi - math.pi / 2


class CGDimer(nn.Module):
    def __init__(self, n, tol):
        super().__init__()
        self.n = n
        self.tol = tol
        self.create_graph = True

    def forward(self, mol: Dict[str, Tensor], create_graph: bool,
                retain_graph: bool):
        mol = mol.copy()
        if p.dim in mol:
            n = normalize(mol[p.dim])
        else:
            n = normalize(mol[p.frc])
        hn = grad(mol[p.frc], mol[p.pos], n)
        nhn = (hn * n).flatten(1).sum(1)
        f = orthogonalized_delta_force(nhn, hn, n)
        g = f
        g_ = normalize(g)
        ghn = (hn * g_).flatten(1).sum(1)
        hg = grad(mol[p.frc], mol[p.pos], g_)
        ghg = (hg * g_).flatten(1).sum(1)
        for _ in range(self.n):
            theta = rotate_angle(nhn, ghn, ghg)[:, None, None]
            n_old = n
            f_old = f
            g_old = g
            g_old_norm = normalize(g_old)
            n = normalize(theta.cos() * n_old + theta.sin() * g_old_norm)
            g_rot = theta.cos() * g_old_norm - theta.sin() * n_old
            hn = grad(mol[p.frc], mol[p.pos], n, allow_unused=False)
            nhn = (n * hn).flatten(1).sum(1)
            f = orthogonalized_delta_force(nhn, hn, n)
            gam = (((f - f_old) * f).flatten(1).sum(1)
                   / (f_old * f_old).flatten(1).sum(1))
            g = f + gam[:, None, None] * g_old.flatten(1).norm(
                p=2, dim=1)[:, None, None] * normalize(g_rot)
            assert (f == f).all()
            # val = f.flatten(1).norm(p=2, dim=1)
            val = f.norm(p=2, dim=2).max()
            print(_, theta * 180 / math.pi)
            if (val < self.tol).all():
                break
        frc_pll = (mol[p.frc] * n).flatten(1).sum(1)[:, None, None] * n
        mol[p.frc] = mol[p.frc] - 2 * frc_pll
        return mol


class Dimer(nn.Module):
    def __init__(self, n, tol):
        super().__init__()
        self.n = n
        self.tol = tol
        self.create_graph = True
        self.c = None

    def forward(self, mol: Dict[str, Tensor], create_graph: bool,
                retain_graph: bool):
        mol = mol.copy()
        num_bch, num_atm, num_dim = mol[p.pos].size()
        num = num_bch * num_atm * num_dim
        assert num_bch == 1
        if p.dim in mol:
            n = normalize(mol[p.dim])
        else:
            n = normalize(mol[p.frc])
        n = n.flatten().detach().numpy()
        def inner(g):
            g_ = torch.from_numpy(g).view((num_bch, num_atm, num_dim))
            ret = -grad(mol[p.frc], mol[p.pos], g_).flatten().detach().numpy()
            return ret
        op = LinearOperator((num, num), inner)
        if self.c is None:
        # if True:
            e, c = eigsh(op, k=1, v0=n, which='SA')
            self.c = c
        else:
            e, c = lobpcg(op, self.c, largest=False)
            self.c = c
        n = c[:, 0]
        n = torch.from_numpy(n).view((num_bch, num_atm, num_dim))
        frc_pll = (mol[p.frc] * n).flatten(1).sum(1)[:, None, None] * n
        mol[p.dim] = n
        mol[p.frc] = mol[p.frc] - 2 * frc_pll
        return mol
