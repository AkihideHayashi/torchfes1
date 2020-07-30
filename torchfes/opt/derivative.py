from typing import Optional
import torch
from torch import Tensor
from ..utils import grad


def _dot(x: Tensor, y: Tensor):
    assert x.dim() == 2
    assert y.dim() == 2
    return (x * y).sum(1)


def directional_derivative(f: Tensor, x: Tensor, d: Tensor, n: int):
    if n == 1:
        g = grad(f, x)
        return _dot(g, d)
    elif n == 2:
        g = grad(f, x, create_graph=True)
        return grad(g, x, grd_out=d)
    else:
        raise NotImplementedError(f'directional_derivative {n}.')


def hessian(f: Tensor, x: Tensor, g: Optional[Tensor] = None):
    if g is None:
        g = grad(f, x, create_graph=True)
    hs = []
    bch, dim = x.size()
    eye = torch.eye(dim, device=x.device, dtype=x.dtype)[None, :, :].expand(
        (bch, -1, -1))
    for i in range(dim):
        hs.append(grad(g, x, eye[:, :, i], retain_graph=True))
    h = torch.stack(hs, dim=2)
    ht = h.transpose(1, 2)
    assert torch.allclose(h, ht, atol=1e-5), (h - ht).abs().max()
    return (h + ht) * 0.5


def jacobian(f: Tensor, x: Tensor):
    if f.size(1) == 0:
        return torch.zeros([f.size(0), f.size(1), x.size(1)],
                           device=x.device, dtype=x.dtype)
    js = []
    bch, out = f.size()
    eye = torch.eye(out, device=x.device, dtype=x.dtype)[None, :, :].expand(
        (bch, -1, -1))
    for i in range(out):
        js.append(grad(f, x, eye[:, :, i], retain_graph=True))
    j = torch.stack(js, dim=1)
    return j
