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
    msk: Tensor
    pbc: Tensor

    def __init__(self, msk: Tensor, pbc: Tensor,
                 wdt: Union[Tensor, List[float]], hgt: Union[Tensor, float],
                 ):
        super().__init__()
        if isinstance(wdt, list):
            wdt = torch.tensor(wdt)
        if isinstance(hgt, float):
            hgt = torch.tensor(hgt)
        self.register_buffer('prc', 1 / (wdt * wdt))
        self.register_buffer('hgt', hgt)
        self.register_buffer('msk', msk)
        self.register_buffer('pbc', pbc)

    def forward(self, inp: Dict[str, Tensor]):
        col = inp[p.col_var][:, self.msk]
        return new_gaussian_mtd(self.prc, self.hgt, col)


class WellTemparedMetaDynamics(nn.Module):
    pbc: Tensor
    msk: Tensor
    prc: Tensor
    hgt: Tensor
    gam: Tensor  # bias_factor

    def __init__(self, msk: Tensor, pbc: Tensor,
                 wdt: Union[Tensor, List[float]], hgt: Union[Tensor, float],
                 gam: Union[Tensor, float]):
        super().__init__()
        if isinstance(wdt, list):
            wdt = torch.tensor(wdt)
        if isinstance(hgt, float):
            hgt = torch.tensor(hgt)
        if isinstance(gam, float):
            gam = torch.tensor(gam)
        self.register_buffer('prc', 1 / (wdt * wdt))
        self.register_buffer('hgt', hgt)
        self.register_buffer('gam', gam)
        self.register_buffer('pbc', pbc)
        self.register_buffer('msk', msk)

    def forward(self, inp: Dict[str, Tensor]):
        col = inp[p.col_var][:, self.msk]
        if p.mtd_cen in inp:
            eng = gaussian_inner(
                self.pbc, inp[p.mtd_prc], inp[p.mtd_cen], inp[p.mtd_hgt], col)
        else:
            eng = torch.zeros_like(inp[p.tim])
        new = new_gaussian_wtmtd(
            self.prc, self.hgt, col, self.gam, inp[p.kbt], eng)
        return new
