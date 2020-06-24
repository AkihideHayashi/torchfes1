from typing import Dict, NamedTuple, Union

import torch
from torch import Tensor, nn

from .. import properties as p


class MTDHillsData(NamedTuple):
    cen: Tensor
    wdt: Tensor
    hgt: Tensor


def gaussian_potential(col, hil: MTDHillsData):
    var = hil.wdt * hil.wdt
    r = hil.cen[:, None, :] - col[None, :, :]  # mtd, bch, col
    v = var[:, None, :]
    h = hil.hgt[:, None]
    ret = (h * torch.exp(-0.5 * (r * r / v).sum(-1))).sum(0)
    return ret


def add_gaussian(hil: MTDHillsData, cen: Tensor, wdt: Tensor, hgt: Tensor):
    return MTDHillsData(
        cen=torch.cat([hil.cen, cen]),
        wdt=torch.cat([hil.wdt, wdt]),
        hgt=torch.cat([hil.hgt, hgt]),
    )


class MTDHills(nn.Module):
    def __init__(self, n_col: int):
        super().__init__()
        self.cen = torch.zeros([0, n_col])
        self.wdt = torch.zeros([0, n_col])
        self.hgt = torch.zeros([0])

    def set_hills(self, hills: MTDHillsData):
        self.cen = hills.cen
        self.wdt = hills.wdt
        self.hgt = hills.hgt

    def forward(self):
        return MTDHillsData(self.cen, self.wdt, self.hgt)


class MetaDynamics(nn.Module):
    """
    Args:
        col: colvar
        hil: MTDHills
        wdt (float[n_col]): width of gaussian
        hgt (float): height of gaussian
    """
    def __init__(self, col, hil: MTDHills, wdt: Tensor,
                 hgt: Union[float, Tensor]):
        if isinstance(hgt, float):
            hgt = torch.tensor(hgt)
        super().__init__()
        self.col = col
        self.hil = hil
        self.wdt = wdt
        self.hgt = hgt

    def forward(self, inp: Dict[str, Tensor]):
        return gaussian_potential(self.col(inp), self.hil())[:, None]

    def append(self, inp: Dict[str, Tensor]):
        col = self.col(inp)
        n_bch = col.size(0)
        wdt = self.wdt[None, :].expand((n_bch, -1))
        hgt = self.hgt[None].expand((n_bch, ))
        self.hil.set_hills(add_gaussian(self.hil(), col, wdt, hgt))


class WellTemparedMetaDynamics(nn.Module):
    """
    Args:
        col: colvar
        hil: MTDHills
        wdt (float[n_col]): width of gaussian
        hgt (float): height of gaussian
        gam (float): gamma
    """
    def __init__(self, col, hil: MTDHills,
                 wdt: Tensor, hgt: Union[float, Tensor],
                 gam: Union[float, Tensor]):
        if isinstance(hgt, float):
            hgt = torch.tensor(hgt)
        if isinstance(gam, float):
            gam = torch.tensor(gam)
        super().__init__()
        self.col = col
        self.hil = hil
        self.wdt = wdt
        self.hgt = hgt
        self.gam = gam

    def forward(self, inp: Dict[str, Tensor]):
        ret = gaussian_potential(self.col(inp), self.hil())
        return ret[:, None]

    def append(self, inp: Dict[str, Tensor]):
        col = self.col(inp)
        n_bch = col.size(0)
        kbt = inp[p.kbt]
        det = self.gam * kbt - kbt
        eng = gaussian_potential(col, self.hil())
        wdt = self.wdt[None, :].expand((n_bch, -1))
        hgt = self.hgt[None].expand((n_bch, )) * torch.exp(-eng / det)
        self.hil.set_hills(add_gaussian(self.hil(), col, wdt, hgt))
        return MTDHillsData(cen=col, wdt=wdt, hgt=hgt)
