"""Quasi-Newton.
yk: g(k+1) - g(k)
sk: x(k+1) - x(k)
Bk: Hessian
rk: 1 / (y(k)^T s(k))
"""
from torch import Tensor


def _vv(x: Tensor, y: Tensor):
    return (x * y).sum(-1)[:, None, None]


def _outer(x: Tensor, y: Tensor):
    return x[:, :, None] * y[:, None, :]


def _mv(m: Tensor, v: Tensor):
    return (m @ v[:, :, None]).squeeze(2)


def bfgs_hes(B: Tensor, s: Tensor, y: Tensor):
    assert B.dim() == 3
    assert y.dim() == 2
    assert s.dim() == 2
    yyT = _outer(y, y)
    yTs = _vv(y, s)
    Bs = _mv(B, s)
    BssTBT = _outer(Bs, Bs)
    sTBs = _vv(s, Bs)
    return B + yyT / yTs - BssTBT / sTBs


def bfgs_inv(B: Tensor, s: Tensor, y: Tensor):
    assert B.dim() == 3
    assert y.dim() == 2
    assert s.dim() == 2
    sTy = _vv(s, y)
    By = _mv(B, y)
    ssT = _outer(s, s)
    yTBy = _vv(y, By)
    BysT = _outer(By, s)
    syTB = BysT.transpose(1, 2)
    den1 = (sTy + yTBy) * ssT
    num1 = sTy * sTy
    den2 = BysT + syTB
    num2 = sTy
    return B + den1 / num1 - den2 / num2


def _dot(x: Tensor, y: Tensor):
    return (x * y).sum(-1)[:, None]


def lbfgs(g: Tensor, s: Tensor, y: Tensor, r: Tensor):
    assert s.dim() == 3  # new, bch, dim
    assert y.dim() == 3  # new, bch, dim
    assert r.dim() == 3  # new, bch, 1
    assert r.size(2) == 1
    q = g  # bch, dim
    k = s.size(2)
    a = []
    for i in range(k - 1, -1, -1):
        ai = r[i] * _dot(s[i], q)
        q = q - ai * y[i]
        a.append(ai)
    a.reverse()
    gamma = _dot(s[-1], y[-1]) / _dot(s[-1], y[-1])  # bch, 1
    z = gamma * q
    for i in range(k):
        bi = r[i] * _dot(y[i], z)
        z = z + s[i] * a[i] - bi
    return -z
