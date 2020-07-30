from typing import List
import torch
from torch import Tensor
from .utils import _vector_pos
from .. import properties as p
from .derivative import jacobian, hessian


def split_hessian(h: Tensor, a: Tensor):
    n_con = a.size(0)
    u_ = _split_dof(a)
    h_ = (u_ @ h @ u_.T)[n_con:, n_con:]
    e, u = torch.symeig(h_, eigenvectors=True)
    return e, (u.T @ u_[n_con:, :]).T


def _split_dof(a: Tensor):
    n_con, n_dim = a.size()
    eye = torch.eye(n_dim)
    u = torch.cat([a, eye])
    return _gram_schmidt(u)


def _gram_schmidt(u: Tensor):
    e: List[Tensor] = []
    _, n = u.size()
    for i in range(u.size(0)):
        new = _normalize(u[i])
        new_ = new.clone()
        for ei in e:
            assert torch.allclose(ei.norm(), torch.tensor(1.0)), ei.norm()
            new -= (new_ @ ei) * ei
        if new.norm() > 1e-6:
            e.append(_normalize(new))
    assert len(e) == n
    return torch.stack(e)


def _normalize(x: Tensor):
    if x.dim() == 1:
        return x / x.norm()
    elif x.dim() == 2:
        return x / x.norm(p=2, dim=1)[:, None]
    else:
        raise NotImplementedError()


def fixed_eig(adj, eng, con, mol):
    assert mol[p.pos].size(0) == 1
    mol = mol.copy()
    q, pos = _vector_pos(mol[p.pos])
    mol[p.pos] = pos
    mol = adj(mol)
    mol = eng(mol)
    col = con(mol)
    a = jacobian(col, q)[0]
    h = hessian(mol[p.eng], q)[0]
    e, c = split_hessian(h, a)
    _, n_atm, n_dim = pos.size()
    c = c.t().view([-1, n_atm, n_dim])
    return e, c
