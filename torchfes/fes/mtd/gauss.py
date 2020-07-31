import math
from typing import Dict
import torch
from torch import Tensor, nn
from ... import properties as p


def get_log_max(tensor: Tensor):
    if tensor.dtype == torch.float16:
        return 11.089866488461016302835560054518282413482666015625
    elif tensor.dtype == torch.float32:
        return 88.7228390520683518616351648233830928802490234375
    elif tensor.dtype == torch.float64:
        return 709.7827128933839730962063185870647430419921875
    else:
        raise TypeError()


def _gaussian_raw_inner(L, n, a, c, h, x):
    #  bch, mtd, dim, pbc
    Ln = torch.where(n == 0, torch.zeros_like(n * L), n * L)
    return (h * (- 0.5 * (x - c - Ln).pow(2) * a).exp().sum(3).prod(2)).sum(1)


def _gaussian_inner(L, a, c, h, x):
    """
    Args:
        L (float[bch, mtd, dim, pbc]): PBC Length. (inf for no pbc.)
        a (float[bch, mtd, dim, pbc]): Precision. (1 / sigma ** 2)
        c (float[bch, mtd, dim, pbc]): Gaussian center.
        h (float[bch, mtd]): Gaussian height.
        x (float[bch, mtd, dim, pbc]): Coordinate.
    """
    # L, a, c, x: bch, mtd, dim, pbc
    # h: bch, mtd
    al2 = a * L * L
    max_n = math.ceil(math.sqrt(2 * get_log_max(al2) / al2.min().item()))
    n = torch.arange(-max_n, max_n + 1, dtype=x.dtype,
                     device=x.device)[None, None, None, :]
    return _gaussian_raw_inner(L, n, a, c, h, x)


def gaussian_inner(L, a, c, h, x):
    """
    Args:
        L (float[dim]): PBC Length. (inf for no pbc.)
        a (float[bch, mtd, dim]): Precision. (1 / sigma ** 2)
        c (float[bch, mtd, dim]): Gaussian center.
        h (float[bch, mtd]): Gaussian height.
        x (float[bch, dim]): Coordinate.
    """
    return _gaussian_inner(L[None, None, :, None], a[:, :, :, None],
                           c[:, :, :, None], h[:, :], x[:, None, :, None]
                           )


def gaussian(hil: Dict[str, Tensor], pbc: Tensor, col: Tensor):
    return gaussian_inner(
        pbc, hil[p.mtd_prc], hil[p.mtd_cen], hil[p.mtd_hgt], col)


class GaussianPotential(nn.Module):
    pbc: Tensor

    def __init__(self, col: nn.Module):
        super().__init__()
        self.col = col
        if not isinstance(col.pbc, Tensor):
            raise KeyError(type(col.pbc))
        self.register_buffer('pbc', col.pbc)

    def forward(self, inp: Dict[str, Tensor]):
        if p.mtd_cen in inp:
            col = self.col(inp)
            return gaussian(inp, self.pbc, col)[:, None]
        else:
            return torch.zeros_like(inp[p.eng_mol])[:, None]
