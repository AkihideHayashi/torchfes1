from typing import Dict
import torch
from torch import Tensor, nn
from .. import properties as p
from .utils import jacobian, hessian
from .utils import Lagrangian


class FixedEig(nn.Module):
    def __init__(self, evl, con):
        super().__init__()
        self.eng = Lagrangian(evl)
        self.con = con

    def forward(self, mol: Dict[str, Tensor]):
        mol[p.gen_pos].requires_grad_()
        mol = self.eng(mol, create_graph=True)
        q = mol[p.gen_pos]
        col = self.con(mol) - mol[p.con_cen]
        at = jacobian(col, q, create_graph=True)
        h = hessian(mol[p.gen_grd], q)
        e, c = split_hessian(h, at)
        n_bch, n_atm, n_dim = mol[p.pos].size()
        n_mod = c.size(1)
        c = c.view([n_bch, n_mod, n_atm, n_dim])
        return e, c


def split_hessian(h: Tensor, at: Tensor):
    # at: bch, dim, col
    n_con = at.size(2)
    if n_con == 0:
        return torch.symeig(h, eigenvectors=True)
    q, r = torch.qr(at, some=False)
    q = q[:, :, n_con:]  # bch, dim, (dim - col)
    qt = q.transpose(1, 2)
    h_ = qt @ h @ q
    e, u = torch.symeig(h_, eigenvectors=True)
    return e, u.transpose(1, 2) @ qt
