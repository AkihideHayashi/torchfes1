from typing import Dict, Union, List
import torch
from torch import Tensor, nn
from ... import properties as p
from .gauss import gaussian_inner


def new_gaussian_mtd(prc: Tensor, hgt: Tensor, col: Tensor):
    n_bch, n_col = col.size()
    assert prc.size() == (n_col, ), prc.size()
    assert hgt.size() == (), hgt.size()
    prc_new = prc[None, :].expand((n_bch, n_col))
    hgt_new = hgt[None].expand((n_bch, ))
    return {p.mtd_cen: col, p.mtd_prc: prc_new, p.mtd_hgt: hgt_new}


def new_gaussian_wtmtd(prc: Tensor, hgt: Tensor, col: Tensor,
                       gam: Tensor, kbt: Tensor, eng: Tensor):
    n_bch, n_col = col.size()
    assert gam.size() == ()
    assert kbt.size() == (n_bch, )
    assert eng.size() == (n_bch, ), eng.size()
    assert prc.size() == (n_col, ), prc.size()
    assert hgt.size() == (), hgt.size()
    prc_new = prc[None, :].expand((n_bch, n_col))
    hgt_new = hgt[None].expand((n_bch, ))
    det = gam * kbt - kbt
    hgt_wt = hgt_new * torch.exp(-eng / det)
    return {p.mtd_cen: col, p.mtd_prc: prc_new, p.mtd_hgt: hgt_wt}


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
        return new_gaussian_mtd(self.prc, self.hgt, self.col(inp))


class WellTemparedMetaDynamics(nn.Module):
    pbc: Tensor
    prc: Tensor
    hgt: Tensor
    gam: Tensor  # bias_factor

    def __init__(self, col: nn.Module,
                 wdt: Union[Tensor, List[float]], hgt: Union[Tensor, float],
                 gam: Union[Tensor, float]):
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
        if not isinstance(self.col.pbc, Tensor):
            raise KeyError(col.pbc)
        self.register_buffer('pbc', self.col.pbc)

    def forward(self, inp: Dict[str, Tensor]):
        col = self.col(inp)
        if p.mtd_cen in inp:
            eng = gaussian_inner(
                self.pbc, inp[p.mtd_prc], inp[p.mtd_cen], inp[p.mtd_hgt], col)
        else:
            eng = torch.zeros_like(inp[p.tim])
        new = new_gaussian_wtmtd(
            self.prc, self.hgt, col, self.gam, inp[p.kbt], eng)
        return new
