import torch
from torch import Tensor


def batch_diagflat(diagonal):
    """Batch diagflat."""
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


def _diag_cat(b: Tensor, F: Tensor):
    """
    b1 ..  0|F1
    .. .. ..|..
     0 .. bm|Fm
    -----------
    F1 .. Fm| 0
    """
    bch = b.size(0)
    assert F.size(0) == bch
    z = torch.zeros([bch, 1, 1], device=F.device, dtype=F.dtype)
    return torch.cat([
        torch.cat([batch_diagflat(b), F[:, :, None]], dim=2),
        torch.cat([F[:, None, :], z], dim=2)], dim=1)


def eigenvector_following(h: Tensor, g: Tensor, n: int):
    """Implementation of eigenvector following.
    See "Geometry Optimization in Cartesian Coordinates:
         Constrained Optimization"
    Jon Baker
    Journal of Computational Chemistry, Vol. 13, No. 2, 240-253 (1992)

    Args:
        h: Hessian
        g: Gradient
        n: The desired number of negative eigen values.
    """
    b, U = torch.symeig(h, eigenvectors=True)
    F = (g[:, None, :] @ U).squeeze(1)
    bF1 = _diag_cat(b[:, :n], F[:, :n])
    bF2 = _diag_cat(b[:, n:], F[:, n:])
    lmd1 = torch.symeig(bF1, eigenvectors=False)[0][:, -1][:, None]
    lmd2 = torch.symeig(bF2, eigenvectors=False)[0][:, 0][:, None]
    stp1 = -(U[:, :, :n] @ (F[:, :n] / (b[:, :n] - lmd1))[:, :, None])
    stp2 = -(U[:, :, n:] @ (F[:, n:] / (b[:, n:] - lmd2))[:, :, None])
    stp = (stp1 + stp2).squeeze(2)
    return stp
