from typing import Dict
import torch
from torch import Tensor
from ...utils import grad
from ... import properties as p


def _dot(x: Tensor, y: Tensor):
    assert x.dim() == 2
    assert y.dim() == 2
    return (x * y).sum(1)


def directional_derivative(g: Tensor, x: Tensor, d: Tensor, n: int):
    if n == 1:
        return _dot(g, d)
    elif n == 2:
        return _dot(d, grad(g, x, grd_out=d, retain_graph=True))
    else:
        raise NotImplementedError(f'directional_derivative {n}.')


def hessian(g: Tensor, x: Tensor):
    hs = []
    bch, dim = x.size()
    eye = torch.eye(dim, device=x.device, dtype=x.dtype)[None, :, :].expand(
        (bch, -1, -1))
    for i in range(dim):
        hs.append(grad(g, x, eye[:, :, i], retain_graph=True))
    h = torch.stack(hs, dim=2)
    ht = h.transpose(1, 2)
    assert torch.allclose(h, ht, atol=1e-4), (h - ht).abs().max()
    return (h + ht) * 0.5


def jacobian(f: Tensor, x: Tensor, create_graph: bool):
    if f.size(1) == 0:
        return torch.zeros([f.size(0), x.size(1), f.size(1)],
                           device=x.device, dtype=x.dtype)
    js = []
    bch, out = f.size()
    eye = torch.eye(out, device=x.device, dtype=x.dtype)[None, :, :].expand(
        (bch, -1, -1))
    for i in range(out):
        js.append(grad(f, x, eye[:, :, i], retain_graph=True,
                       create_graph=create_graph))
    j = torch.stack(js, dim=-1)
    return j


def set_hessian(mol: Dict[str, Tensor]):
    mol = mol.copy()
    mol[p.gen_hes] = hessian(mol[p.gen_grd], mol[p.gen_pos])
    return mol


def set_directional_gradient(mol: Dict[str, Tensor]):
    mol = mol.copy()
    mol[p.gen_dir_grd] = directional_derivative(
        mol[p.gen_grd], mol[p.gen_pos], mol[p.gen_stp_dir], 1)
    return mol


def set_directional_hessian(mol: Dict[str, Tensor]):
    mol = mol.copy()
    mol[p.gen_dir_hes] = directional_derivative(
        mol[p.gen_grd], mol[p.gen_pos], mol[p.gen_stp_dir], 2)
    return mol
