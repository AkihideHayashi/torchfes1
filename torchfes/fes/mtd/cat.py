from typing import Dict
import torch
from torch import nn, Tensor
from ... import properties as p


def cat_gaussian(inp: Dict[str, Tensor], new: Dict[str, Tensor]):
    out = inp.copy()
    if p.mtd_cen in inp:
        out[p.mtd_cen] = torch.cat([inp[p.mtd_cen], new[p.mtd_cen]], dim=1)
        out[p.mtd_prc] = torch.cat([inp[p.mtd_prc], new[p.mtd_prc]], dim=1)
        out[p.mtd_hgt] = torch.cat([inp[p.mtd_hgt], new[p.mtd_hgt]], dim=1)
    else:
        assert p.mtd_hgt not in inp
        assert p.mtd_prc not in inp
        out[p.mtd_cen] = new[p.mtd_cen]
        out[p.mtd_prc] = new[p.mtd_prc]
        out[p.mtd_hgt] = new[p.mtd_hgt]
    return out


class BatchMTD(nn.Module):
    def __init__(self, new):
        super().__init__()
        self.new = new

    def forward(self, mol: Dict[str, Tensor]):
        tmp: Dict[str, Tensor] = self.new(mol)
        cen = tmp[p.mtd_cen]  # bch, col
        prc = tmp[p.mtd_prc]  # bch, col
        hgt = tmp[p.mtd_hgt]  # bch,
        new = {
            p.mtd_cen: cen[:, None, :],
            p.mtd_prc: prc[:, None, :],
            p.mtd_hgt: hgt[:, None],
            p.idt: mol[p.idt],
        }
        out = cat_gaussian(mol, new)
        return out, new


class EnsembleMTD(nn.Module):
    def __init__(self, new):
        super().__init__()
        self.new = new

    def forward(self, mol: Dict[str, Tensor]):
        tmp: Dict[str, Tensor] = self.new(mol)
        cen = tmp[p.mtd_cen]  # bch, col
        prc = tmp[p.mtd_prc]  # bch, col
        hgt = tmp[p.mtd_hgt]  # bch,
        new = {
            p.mtd_cen: cen[None, :, :],
            p.mtd_prc: prc[None, :, :],
            p.mtd_hgt: hgt[None, :]
        }
        out = cat_gaussian(mol, new)
        return out, new
