"""Conjugate gradient linear equation solver."""
from typing import NamedTuple

import torch
from torch import Tensor


def conjugate_gradient(A: Tensor, b: Tensor, x0: Tensor, eps: Tensor):
    """Reference impelementation how to use this module."""
    A = A.detach()
    b = b.detach()
    x = x0.detach()
    Ax = A @ x
    stp = init_conjugate_gradient_step(x=x, Ax=Ax, b=b, eps=eps)
    while stp.mask.any():
        Ap = A @ stp.p
        stp = conjugate_gradient_step(stp, Ap, eps)
    return stp.x


class ConjugateGradientStep(NamedTuple):
    x: Tensor
    r: Tensor
    p: Tensor
    mask: Tensor
    rmag: Tensor


def init_conjugate_gradient_step(x: Tensor, Ax: Tensor, b: Tensor,
                                 eps: Tensor):
    r = Ax - b
    p = -r
    return make_conjugate_gradient_step(x, r, p, eps)


def make_conjugate_gradient_step(x: Tensor, r: Tensor, p: Tensor, eps: Tensor):
    assert x.dim() == 3
    assert p.dim() == 3
    assert r.dim() == 3
    rmag = r.pow(2).sum(dim=1, keepdim=True)
    mask = rmag >= eps[:, None, None]
    return ConjugateGradientStep(x=x, r=r, p=p, mask=mask, rmag=rmag)


def conjugate_gradient_step(stp: ConjugateGradientStep, Ap: Tensor,
                            eps: Tensor):
    rmag = stp.rmag
    filt = stp.mask
    z = torch.zeros_like(rmag)
    alph = torch.where(filt, rmag / (stp.p * Ap).sum(dim=1, keepdim=True), z)
    x = stp.x + alph * stp.p
    r = stp.r + alph * Ap
    beta = torch.where(filt, r.pow(2).sum(dim=1, keepdim=True) / rmag, z)
    p = -r + beta * stp.p
    return make_conjugate_gradient_step(x=x, r=r, p=p, eps=eps)
