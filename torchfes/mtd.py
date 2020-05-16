from typing import Dict, Optional, NamedTuple
import torch
from torch import nn, Tensor
from . import properties as p


# class MetaDynamics(nn.Module):
#     def __init__(self, col,
#                  w: Tensor, h: Tensor,
#                  cen: Optional[Tensor] = None,
#                  wdt: Optional[Tensor] = None,
#                  hgt: Optional[Tensor] = None):
#         super().__init__()
#         n_dim = col.n_dim
#         self.col = col
#         self.w = w
#         self.h = h
#         if cen is None:
#             assert wdt is None and hgt is None
#             self.cen = torch.zeros([0, n_dim])
#             self.wdt = torch.zeros([0, n_dim])
#             self.hgt = torch.zeros([0])
#         else:
#             assert wdt is not None and hgt is not None
#             self.cen = cen
#             self.wdt = wdt
#             self.hgt = hgt

#     def forward(self, inp: Dict[str, Tensor]):
#         col = self.col(inp)
#         assert col.size(1) == self.col.n_dim
#         return gaussian_potential(col, self.cen, self.wdt, self.hgt)

#     def append(self, inp: Dict[str, Tensor]):
#         col = self.col(inp)
#         n_bch = col.size(0)
#         wdt = self.w[None, :].expand((n_bch, -1))
#         hgt = self.h[None].expand((n_bch, ))
#         self.cen, self.wdt, self.hgt = add_gaussian(
#             self.cen, self.wdt, self.hgt, col, wdt, hgt)


# class WellTemparedMetaDynamics(nn.Module):
#     def __init__(self, col,
#                  w: Tensor, h: Tensor, g: Tensor,
#                  cen: Optional[Tensor] = None,
#                  wdt: Optional[Tensor] = None,
#                  hgt: Optional[Tensor] = None):
#         super().__init__()
#         n_dim = col.n_dim
#         self.col = col
#         self.w = w
#         self.h = h
#         self.gam = g
#         if cen is None:
#             assert wdt is None and hgt is None
#             self.cen = torch.zeros([0, n_dim])
#             self.wdt = torch.zeros([0, n_dim])
#             self.hgt = torch.zeros([0])
#         else:
#             assert wdt is not None and hgt is not None
#             self.cen = cen
#             self.wdt = wdt
#             self.hgt = hgt

#     def forward(self, inp: Dict[str, Tensor]):
#         col = self.col(inp)
#         assert col.size(1) == self.col.n_dim
#         return gaussian_potential(col, self.cen, self.wdt, self.hgt)

#     def append(self, inp: Dict[str, Tensor]):
#         col = self.col(inp)
#         n_bch = col.size(0)
#         kbt = inp[p.kbt]
#         det = self.gam * kbt - kbt
#         eng = gaussian_potential(col, self.cen, self.wdt, self.hgt)
#         wdt = self.w[None, :].expand((n_bch, -1))
#         hgt = self.h[None].expand((n_bch, )) * torch.exp(-eng / det)
#         self.cen, self.wdt, self.hgt = add_gaussian(
#             self.cen, self.wdt, self.hgt, col, wdt, hgt)


# def add_gaussian(cen: Tensor, wdt: Tensor, hgt: Tensor,
#                  c: Tensor, w: Tensor, h: Tensor):
#     return (
#         torch.cat([cen, c]),
#         torch.cat([wdt, w]),
#         torch.cat([hgt, h])
#     )

# def gaussian_potential(col, cen, wdt, hgt):
#     var = wdt * wdt
#     r = cen[:, None, :] - col[None, :, :]  # mtd, bch, col
#     v = var[:, None, :]
#     h = hgt[:, None]
#     return (h * torch.exp(-0.5 * (r * r / v).sum(-1))).sum(0)


class MTDHillsData(NamedTuple):
    cen: Tensor
    wdt: Tensor
    hgt: Tensor


def gaussian_potential(col, hil: MTDHillsData):
    var = hil.wdt * hil.wdt
    r = hil.cen[:, None, :] - col[None, :, :]  # mtd, bch, col
    v = var[:, None, :]
    h = hil.hgt[:, None]
    return (h * torch.exp(-0.5 * (r * r / v).sum(-1))).sum(0)


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
    def __init__(self, col, hil: MTDHills, wdt: Tensor, hgt: Tensor):
        super().__init__()
        self.col = col
        self.hil = hil
        self.wdt = wdt
        self.hgt = hgt

    def forward(self, inp: Dict[str, Tensor]):
        return gaussian_potential(self.col(inp), self.hil())

    def append(self, inp: Dict[str, Tensor]):
        col = self.col(inp)
        n_bch = col.size(0)
        wdt = self.wdt[None, :].expand((n_bch, -1))
        hgt = self.hgt[None].expand((n_bch, ))
        self.hil.set_hills(add_gaussian(self.hil(), col, wdt, hgt))


class WellTemparedMetaDynamics(nn.Module):
    def __init__(self, col, hil: MTDHills,
                 wdt: Tensor, hgt: Tensor, gam: Tensor):
        super().__init__()
        self.col = col
        self.hil = hil
        self.wdt = wdt
        self.hgt = hgt
        self.gam = gam

    def forward(self, inp: Dict[str, Tensor]):
        return gaussian_potential(self.col(inp), self.hil())

    def append(self, inp: Dict[str, Tensor]):
        col = self.col(inp)
        n_bch = col.size(0)
        kbt = inp[p.kbt]
        det = self.gam * kbt - kbt
        eng = gaussian_potential(col, self.hil())
        wdt = self.wdt[None, :].expand((n_bch, -1))
        hgt = self.hgt[None].expand((n_bch, )) * torch.exp(-eng / det)
        self.hil.set_hills(add_gaussian(self.hil(), col, wdt, hgt))
