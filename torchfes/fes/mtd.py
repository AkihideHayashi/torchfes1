import math
from typing import Dict, Union, List
import torch
from torch import Tensor, nn
from .. import properties as p


def get_log_max(tensor: Tensor):
    if tensor.dtype == torch.float16:
        return 11.089866488461016302835560054518282413482666015625
    elif tensor.dtype == torch.float32:
        return 88.7228390520683518616351648233830928802490234375
    elif tensor.dtype == torch.float64:
        return 709.7827128933839730962063185870647430419921875
    else:
        raise TypeError()


def add_gaussian(inp: Dict[str, Tensor], new: Dict[str, Tensor]):
    out = inp.copy()
    if p.mtd_cen in inp:
        out[p.mtd_cen] = torch.cat([inp[p.mtd_cen], new[p.mtd_cen]])
        out[p.mtd_prc] = torch.cat([inp[p.mtd_prc], new[p.mtd_prc]])
        out[p.mtd_hgt] = torch.cat([inp[p.mtd_hgt], new[p.mtd_hgt]])
    else:
        assert p.mtd_hgt not in inp
        assert p.mtd_prc not in inp
        out[p.mtd_cen] = new[p.mtd_cen]
        out[p.mtd_prc] = new[p.mtd_prc]
        out[p.mtd_hgt] = new[p.mtd_hgt]
    return out


def get_gaussian_hills(inp: Dict[str, Tensor]):
    cen = inp[p.mtd_cen]
    prc = inp[p.mtd_prc]
    hgt = inp[p.mtd_hgt]
    return {
        p.mtd_cen: cen,
        p.mtd_prc: prc,
        p.mtd_hgt: hgt,
    }


def gaussian_potential_nopbc_restrict(col: Tensor, hil: Dict[str, Tensor]):
    r = hil[p.mtd_cen][:, None, :] - col[None, :, :]  # mtd, bch, col
    v = hil[p.mtd_prc][:, None, :]
    h = hil[p.mtd_hgt][:, None]
    ret = (h * torch.exp(-0.5 * (r * r * v).sum(-1))).sum(0)
    return ret


def gaussian_potential_pbc_restrict(
        col: Tensor, pbc: Tensor, hil: Dict[str, Tensor]):
    a = hil[p.mtd_prc][None, :, :, None]
    L = pbc[None, None, :, None]
    al2 = a * L * L
    max_n = math.ceil(math.sqrt(2 * get_log_max(al2) / al2.max().item()))
    n = torch.arange(-max_n, max_n + 1, dtype=col.dtype, device=col.device
                     )[None, None, None, :]
    h = hil[p.mtd_hgt][None, :]
    x = col[:, None, :, None]
    c = hil[p.mtd_cen][None, :, :, None]
    nL = torch.where(n == 0, torch.zeros_like(n * L), n * L)
    return (torch.exp(-0.5 * a * (x - c - nL) ** 2).sum(-1).prod(-1) * h
            ).sum(-1)


def gaussian_potential(col: Tensor, pbc: Tensor, hil: Dict[str, Tensor]):
    if p.mtd_hgt not in hil:
        return torch.zeros([col.size(0)], device=col.device, dtype=col.dtype)
    neg_log_eps = get_log_max(col)
    if hil[p.mtd_prc].dim() == 2:
        if (pbc >= math.exp(neg_log_eps)).all():
            ret = gaussian_potential_nopbc_restrict(col, hil)
            assert torch.allclose(
                ret, gaussian_potential_pbc_restrict(col, pbc, hil))
            return ret
        else:
            return gaussian_potential_pbc_restrict(col, pbc, hil)
    else:
        raise NotImplementedError()


def new_gaussian_mtd(prc: Tensor, hgt: Tensor, col: Tensor):
    n_bch, n_col = col.size()
    assert prc.size(0) == n_col
    assert hgt.size() == ()
    prc = prc[None, :].expand((n_bch, -1))
    hgt = hgt[None].expand((n_bch, ))
    return {
        p.mtd_cen: col,
        p.mtd_prc: prc,
        p.mtd_hgt: hgt,
    }


def new_gaussian_wtmtd(prc: Tensor, hgt: Tensor, gam: Tensor,
                       col: Tensor, pbc: Tensor, inp: Dict[str, Tensor]):
    n_bch, n_col = col.size()
    assert prc.size(0) == n_col
    assert hgt.size() == ()
    assert gam.size() == ()
    kbt = inp[p.kbt]
    det = gam * kbt - kbt
    eng = gaussian_potential(col, pbc, inp)
    prc = prc[None, :].expand((n_bch, -1))
    hgt = hgt[None].expand((n_bch, )) * torch.exp(-eng / det)
    return {
        p.mtd_cen: col,
        p.mtd_prc: prc,
        p.mtd_hgt: hgt,
    }


def wtmtd_to_mtd(hil: Dict[str, Tensor], gam: float):
    out = hil.copy()
    out[p.mtd_hgt] = hil[p.mtd_hgt] * (gam / (gam - 1))
    return out


def mtd_to_wtmtd(hil: Dict[str, Tensor], gam: float):
    out = hil.copy()
    out[p.mtd_hgt] = hil[p.mtd_hgt] * ((gam - 1) / gam)
    return out


class GaussianPotential(nn.Module):
    pbc: Tensor

    def __init__(self, col: nn.Module):
        super().__init__()
        self.col = col
        if not isinstance(col.pbc, Tensor):
            raise KeyError(type(col.pbc))
        self.register_buffer('pbc', col.pbc)

    def forward(self, inp: Dict[str, Tensor]):
        col = self.col(inp)
        return gaussian_potential(col, self.pbc, inp)[:, None]


class MetaDynamics(nn.Module):
    prc: Tensor
    hgt: Tensor

    def __init__(self, col: nn.Module,
                 wdt: Union[Tensor, List[float]], hgt: Union[Tensor, float],
                 ):
        super().__init__()
        self.col = col
        if isinstance(wdt, list):
            wdt = torch.tensor(wdt)
        if isinstance(hgt, float):
            hgt = torch.tensor(hgt)
        self.register_buffer('prc', 1 / (wdt * wdt))
        self.register_buffer('hgt', hgt)

    def forward(self, inp: Dict[str, Tensor]):
        new = new_gaussian_mtd(self.prc, self.hgt, self.col(inp))
        out = add_gaussian(inp, new)
        return out, new


class WellTemparedMetaDynamics(nn.Module):
    pbc: Tensor
    prc: Tensor
    hgt: Tensor
    gam: Tensor

    def __init__(self, col: nn.Module,
                 wdt: Union[Tensor, List[float]], hgt: Union[Tensor, float],
                 gam: Union[Tensor, float], new_mtd: bool):
        super().__init__()
        self.col = col
        if isinstance(wdt, list):
            wdt = torch.tensor(wdt)
        if isinstance(hgt, float):
            hgt = torch.tensor(hgt)
        if isinstance(gam, float):
            gam = torch.tensor(gam)
        self.register_buffer('prc', 1 / (wdt * wdt))
        self.register_buffer('hgt', hgt)
        self.register_buffer('gam', gam)
        self.new_mtd = new_mtd
        if not isinstance(self.col.pbc, Tensor):
            raise KeyError(col.pbc)
        self.register_buffer('pbc', self.col.pbc)

    def forward(self, inp: Dict[str, Tensor]):
        new = new_gaussian_wtmtd(
            self.prc, self.hgt, self.gam, self.col(inp), self.pbc, inp)
        out = add_gaussian(inp, new)
        if self.new_mtd:
            new = wtmtd_to_mtd(new, self.gam.item())
        return out, new
