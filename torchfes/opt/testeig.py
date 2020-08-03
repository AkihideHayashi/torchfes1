from typing import Dict
import torch
from torch import Tensor, nn
from .. import properties as p
from .derivative import jacobian, hessian
from .generalize import Generalize


class FixedEig(nn.Module):
    def __init__(self, adj, eng, con):
        super().__init__()
        self.adj = adj
        self.eng = eng
        self.con = con
        self.gen = Generalize(None, None, False)

    def forward(self, mol: Dict[str, Tensor]):
        mol = self.gen(mol)
        mol = self.adj(mol)
        mol = self.eng(mol)
        col = self.con(mol) - mol[p.con_cen]
        q = mol[p.gen_pos]
        a = jacobian(col, q)
        h = hessian(mol[p.eng], q)
        e, c = split_hessian(h, a)
        n_bch, n_atm, n_dim = mol[p.pos].size()
        n_mod = c.size(1)
        c = c.view([n_bch, n_mod, n_atm, n_dim])
        return e, c


def split_hessian(h: Tensor, a: Tensor):
    n_con = a.size(1)
    at = a.transpose(1, 2)  # bch, dim, col
    q, r = torch.qr(at, some=False)
    q = q[:, :, n_con:]  # bch, dim, (dim - col)
    qt = q.transpose(1, 2)
    h_ = qt @ h @ q
    e, u = torch.symeig(h_, eigenvectors=True)
    return e, u.transpose(1, 2) @ qt
