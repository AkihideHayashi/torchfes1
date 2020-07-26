from typing import Dict, Union
from torch import Tensor
from ... import properties as p


def wtmtd_to_mtd(mol: Dict[str, Tensor], gam: Union[float, Tensor]):
    out = mol.copy()
    out[p.mtd_hgt] = mol[p.mtd_hgt] * (gam / (gam - 1))
    return out


def mtd_to_wtmtd(mol: Dict[str, Tensor], gam: Union[float, Tensor]):
    out = mol.copy()
    out[p.mtd_hgt] = mol[p.mtd_hgt] * ((gam - 1) / gam)
    return out
