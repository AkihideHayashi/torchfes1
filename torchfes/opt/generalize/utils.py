from typing import Dict, Optional
from torch import Tensor
import torch
from ... import properties as p


def _split(x: Tensor, n: int):
    return x[:, :n], x[:, n:]


def _cat(x: Tensor, y: Tensor):
    return torch.cat([x, y], dim=1)


def _generalize_pos(mol: Dict[str, Tensor], msk: Optional[Tensor]):
    if msk is None:
        return mol[p.pos].flatten(1)
    else:
        return mol[p.pos][:, msk]


def _specialize_pos(
        mol: Dict[str, Tensor], msk: Optional[Tensor], gen: Tensor):
    out = mol.copy()
    if msk is None:
        out[p.pos] = gen.view_as(mol[p.pos])
    else:
        out[p.pos] = out[p.pos].masked_scatter(msk, gen)
    return out


def _specialize_frc(
        mol: Dict[str, Tensor], msk: Optional[Tensor], gen: Tensor):
    out = mol.copy()
    if msk is None:
        out[p.frc] = gen.view_as(mol[p.pos])
    else:
        out[p.frc] = out[p.pos].masked_scatter(msk, gen).masked_fill(~msk, 0.0)
    return out


def _generalize_add_cel(mol: Dict[str, Tensor], gen):
    cel = mol[p.cel].flatten(1)
    return _cat(gen, cel)


def _specialize_del_cel(mol: Dict[str, Tensor], gen: Tensor):
    num_dim = mol[p.cel].size(2)
    out = mol.copy()
    num_two = num_dim * num_dim
    cel, gen = _split(gen, num_two)
    out[p.cel] = cel.view((-1, num_dim, num_dim))
    return out, gen


def _specialize_del_sts(mol: Dict[str, Tensor], gen: Tensor):
    num_dim = mol[p.cel].size(2)
    out = mol.copy()
    num_two = num_dim * num_dim
    tmp, gen = _split(gen, num_two)
    cel_grd = tmp.view((-1, num_dim, num_dim))
    cel_det = torch.det(out[p.cel])
    out[p.sts] = (cel_grd @ mol[p.cel]) / cel_det[:, None, None]
    return out, gen


def _generalize_add_mul(mol: Dict[str, Tensor], gen: Tensor):
    lam = mol[p.con_mul]
    return _cat(lam, gen)


def _specialize_del_mul(mol: Dict[str, Tensor], gen: Tensor):
    out = mol.copy()
    num_con = mol[p.con_cen].size(1)
    lam, gen = _split(gen, num_con)
    out[p.con_mul] = lam
    return out, gen


def _specialize_del_mul_frc(mol: Dict[str, Tensor], gen: Tensor):
    out = mol.copy()
    num_con = mol[p.con_cen].size(1)
    lam, gen = _split(gen, num_con)
    out[p.con_mul_frc] = lam
    return out, gen


def generalize(mol: Dict[str, Tensor], use_cel: bool, con: bool):
    if p.fix_msk in mol:
        msk: Optional[Tensor] = mol[p.fix_msk]
    else:
        msk = None
    gen = _generalize_pos(mol, msk)
    if use_cel:
        gen = _generalize_add_cel(mol, gen)
    if con is not None:
        gen = _generalize_add_mul(mol, gen)
    ret = mol.copy()
    ret[p.gen_pos] = gen
    return ret


def specialize_pos(mol: Dict[str, Tensor], use_cel: bool, con: bool):
    if p.fix_msk in mol:
        msk: Optional[Tensor] = mol[p.fix_msk]
    else:
        msk = None
    gen = mol[p.gen_pos]
    if con:
        mol, gen = _specialize_del_mul(mol, gen)
    if use_cel:
        mol, gen = _specialize_del_cel(mol, gen)
    mol = _specialize_pos(mol, msk, gen)
    return mol


def specialize_grd(mol: Dict[str, Tensor], use_cel: bool, con: bool):
    if p.fix_msk in mol:
        msk: Optional[Tensor] = mol[p.fix_msk]
    else:
        msk = None
    gen = - mol[p.gen_grd]
    if con:
        mol, gen = _specialize_del_mul_frc(mol, gen)
    if use_cel:
        mol, gen = _specialize_del_sts(mol, gen)
    mol = _specialize_frc(mol, msk, gen)
    return mol
