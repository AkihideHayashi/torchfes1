from typing import Dict, List, Optional, Set
import torch
from torch import Tensor, nn
from .torch import pad as _pad, size_max
from .. import properties as p
from ..properties import default_values, batch, atoms, metad


DT = Dict[str, Tensor]
DLT = Dict[str, List[Tensor]]
DLOT = Dict[str, List[Optional[Tensor]]]
LDT = List[Dict[str, Tensor]]


def _len_dt(dt: DT):
    n = 0
    for key, val in dt.items():
        assert key in batch
        if n in (0, 1):
            n = val.size(0)
        else:
            assert val.size(0) in (n, 1)
    return n


def dt_to_dlt(dt: DT):
    new: DLT = {}
    n = _len_dt(dt)
    for key, val in dt.items():
        assert key in batch
        if val.size(0) == 1:
            new[key] = [val for _ in range(n)]
        else:
            new[key] = list(dt[key].unsqueeze(0).unbind(1))
    return new


def _len_dlt(dlt: DLT):
    n = 0
    for key, val in dlt.items():
        if n == 0:
            n = len(val)
        else:
            assert n == len(val), key
    return n


def dlt_to_ldt(dlt: DLT):
    n = _len_dlt(dlt)
    new: LDT = [{} for _ in range(n)]
    for key in dlt:
        for ne, mo in zip(new, dlt[key]):
            ne[key] = mo
    return new


def _keys_ldt(ldt: LDT):
    keys: Set[str] = set()
    for mo in ldt:
        keys.update(set(mo.keys()))
    return keys


def ldt_to_dlot(ldt: LDT):
    keys = _keys_ldt(ldt)
    for key in keys:
        assert key in batch, key
    new: DLOT = {key: [] for key in keys}
    for mo in ldt:
        for key in keys:
            if key in mo:
                new[key].append(mo[key])
            else:
                new[key].append(None)
    return new


def dlot_to_dlt(dlot: DLOT) -> DLT:
    dlt: DLT = {}
    for key, lot in dlot.items():
        dlt[key] = []
        for ot in lot:
            if ot is None:
                raise RuntimeError(key)
            else:
                dlt[key].append(ot)
    return dlt


def dlt_to_dt(dlt: DLT):
    dt: DT = {}
    for key, lt in dlt.items():
        dt[key] = torch.cat(lt, dim=0)
    return dt


def filter_case(mol: Dict[str, Tensor], case: Set[str]):
    return {key: val for key, val in mol.items() if key in case}


def pak_atm(mol: Dict[str, Tensor]):
    if p.ent not in mol:
        return mol
    ret: Dict[str, Tensor] = {}
    ent = mol[p.ent].squeeze(0)
    assert ent.dim() == 1
    for key in mol:
        assert key in batch
        assert mol[key].size(0) == 1
        if key in atoms:
            ret[key] = mol[key][:, ent]
        else:
            ret[key] = mol[key]
    return ret


def pak_mtd(mol: Dict[str, Tensor]):
    if p.mtd_hgt not in mol:
        return mol
    assert mol[p.mtd_hgt].dim() == 2
    assert mol[p.mtd_cen].dim() == 3
    assert mol[p.mtd_prc].dim() == 3
    assert mol[p.mtd_hgt].size(0) == 1
    mask = mol[p.mtd_hgt].squeeze(0) != 0
    ret = mol.copy()
    ret[p.mtd_hgt] = mol[p.mtd_hgt][:, mask]
    ret[p.mtd_cen] = mol[p.mtd_cen][:, mask, :]
    ret[p.mtd_prc] = mol[p.mtd_prc][:, mask, :]
    return ret


class PakMtd(nn.Module):
    def forward(self, dt: DT):
        return pak_mtd(dt)


class PakAtm(nn.Module):
    def forward(self, dt: DT):
        return pak_atm(dt)


def pad_atm(dlt: DLT):
    new: DLT = {}
    for key, lt in dlt.items():
        if key in atoms:
            new[key] = _pad(lt, default_values[key], dim=0)
        else:
            new[key] = lt
    return new


def pad_mtd(dlt: DLT):
    new: DLT = {}
    for key, lt in dlt.items():
        if key in metad:
            new[key] = _pad(lt, default_values[key], dim=0)
        else:
            new[key] = lt
    return new


class PadAtm(nn.Module):
    def forward(self, dlt: DLT):
        return pad_atm(dlt)


class PadMtd(nn.Module):
    def forward(self, dlt: DLT):
        return pad_mtd(dlt)


class FilMtd(nn.Module):
    def forward(self, dlot: DLOT):
        if not dlot:
            return dlot
        tmp = dlot[p.pos][0]
        assert tmp is not None
        device = tmp.device
        new: DLOT = {}
        for key, lot in dlot.items():
            if key in metad:
                lt: List[Tensor] = []
                for ot_ in lot:
                    if ot_ is not None:
                        lt.append(ot_)
                size = size_max(lt)
                n = []
                for ot in lot:
                    if ot is None:
                        _tmp: Optional[Tensor] = torch.ones(
                            size, device=device) * default_values[key]
                        n.append(_tmp)
                    else:
                        n.append(ot)
                new[key] = n
            else:
                new[key] = lot
        return new


class Cat(nn.Module):
    def __init__(self, prepreprocesses: List[nn.Module],
                 preprocesses: List[nn.Module]):
        super().__init__()
        self.ppp = nn.ModuleList(prepreprocesses)
        self.pp = nn.ModuleList(preprocesses)

    def forward(self, ldt: LDT):
        dlot = ldt_to_dlot(ldt)
        for ppp in self.ppp:
            dlot = ppp(dlot)
        dlt = dlot_to_dlt(dlot)
        for pp in self.pp:
            dlt = pp(dlt)
        dt = dlt_to_dt(dlt)
        return dt


class Unbind(nn.Module):
    def __init__(self, postprecesses: List[nn.Module]):
        super().__init__()
        self.pp = nn.ModuleList(postprecesses)

    def forward(self, dt: DT):
        dt = filter_case(dt, batch)
        dlt = dt_to_dlt(dt)
        ldt = dlt_to_ldt(dlt)
        new = []
        for dt in ldt:
            for pp in self.pp:
                dt = pp(dt)
            new.append(dt)
        return new


def masked_select(mol: List[Dict[str, Tensor]], mask: Tensor):
    assert len(mol) == mask.size(0)
    assert mask.dim() == 1
    return [m for m, s in zip(mol, mask.tolist()) if s]


unbind = Unbind([PakAtm(), PakMtd()])
cat = Cat([FilMtd()], [PadAtm(), PadMtd()])
