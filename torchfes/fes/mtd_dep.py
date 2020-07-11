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
    # mtd_dep_cen: [bch, mtd, col], [bch, col]
    # mtd_dep_prc: [bch, mtd, col], [bch, col]
    # mtd_dep_hgt: [bch, mtd] [bch]
    out = inp.copy()
    if p.mtd_dep_cen in inp:
        out[p.mtd_dep_cen] = torch.cat([inp[p.mtd_dep_cen],
                                        new[p.mtd_dep_cen][:, None, :]], dim=1)
        out[p.mtd_dep_prc] = torch.cat([inp[p.mtd_dep_prc],
                                        new[p.mtd_dep_prc][:, None, :]], dim=1)
        out[p.mtd_dep_hgt] = torch.cat([inp[p.mtd_dep_hgt],
                                        new[p.mtd_dep_hgt][:, None]], dim=1)
    else:
        assert p.mtd_dep_hgt not in inp
        assert p.mtd_dep_prc not in inp
        out[p.mtd_dep_cen] = new[p.mtd_dep_cen][:, None, :]
        out[p.mtd_dep_prc] = new[p.mtd_dep_prc][:, None, :]
        out[p.mtd_dep_hgt] = new[p.mtd_dep_hgt][:, None]
    return out


def get_gaussian_hills(inp: Dict[str, Tensor]):
    cen = inp[p.mtd_dep_cen]
    prc = inp[p.mtd_dep_prc]
    hgt = inp[p.mtd_dep_hgt]
    return {
        p.mtd_dep_cen: cen,
        p.mtd_dep_prc: prc,
        p.mtd_dep_hgt: hgt,
    }


def gaussian_potential(col: Tensor, hil: Dict[str, Tensor]):
    # cen [bch, mtd, col]
    # prc [bch, mtd, col]
    # hgt [bch, mtd]
    assert col.dim() == 2
    assert hil[p.mtd_dep_cen].dim() == 3, hil[p.mtd_dep_cen].size()
    assert hil[p.mtd_dep_prc].dim() == 3
    assert hil[p.mtd_dep_hgt].dim() == 2
    r = hil[p.mtd_dep_cen][:, :, :] - col[:, None, :]  # bch, mtd, col
    v = hil[p.mtd_dep_prc][:, :, :]
    h = hil[p.mtd_dep_hgt][:, :]
    ret = (h * torch.exp(-0.5 * (r * r * v).sum(-1))).sum(1)
    return ret


def new_gaussian_mtd(prc: Tensor, hgt: Tensor, col: Tensor):
    n_bch, n_col = col.size()
    assert prc.size(0) == n_col
    assert hgt.size() == ()
    prc = prc[None, :].expand((n_bch, -1))
    hgt = hgt[None].expand((n_bch, ))
    return {
        p.mtd_dep_cen: col,
        p.mtd_dep_prc: prc,
        p.mtd_dep_hgt: hgt,
    }


def new_gaussian_wtmtd(prc: Tensor, hgt: Tensor, gam: Tensor,
                       col: Tensor, pbc: Tensor, inp: Dict[str, Tensor]):
    n_bch, n_col = col.size()
    assert prc.size(0) == n_col
    assert hgt.size() == ()
    assert gam.size() == ()
    kbt = inp[p.kbt]
    det = gam * kbt - kbt
    eng = gaussian_potential(col, inp)
    prc = prc[None, :].expand((n_bch, -1))
    hgt = hgt[None].expand((n_bch, )) * torch.exp(-eng / det)
    return {
        p.mtd_dep_cen: col,
        p.mtd_dep_prc: prc,
        p.mtd_dep_hgt: hgt,
    }


def wtmtd_to_mtd(hil: Dict[str, Tensor], gam: float):
    out = hil.copy()
    out[p.mtd_dep_hgt] = hil[p.mtd_dep_hgt] * (gam / (gam - 1))
    return out


def mtd_to_wtmtd(hil: Dict[str, Tensor], gam: float):
    out = hil.copy()
    out[p.mtd_dep_hgt] = hil[p.mtd_dep_hgt] * ((gam - 1) / gam)
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
        if p.mtd_dep_cen not in inp:
            return torch.zeros_like(col)[:, 0][:, None]
        return gaussian_potential(col, inp)[:, None]


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
        return out


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
        return out
