from typing import Dict, Optional
from torch import Tensor
import torch
from ... import properties as p


def split(x: Tensor, n: int):
    return x[:, :n], x[:, n:]


def cat(x: Tensor, y: Tensor):
    return torch.cat([x, y], dim=1)


def generalize_pos(mol: Dict[str, Tensor], msk: Optional[Tensor]):
    if msk is None:
        return mol[p.pos].flatten(1)
    else:
        return mol[p.pos][:, msk]


def specialize_pos(mol: Dict[str, Tensor], msk: Optional[Tensor], gen: Tensor):
    out = mol.copy()
    if msk is None:
        out[p.pos] = gen.view_as(mol[p.pos])
    else:
        out[p.pos] = out[p.pos].masked_scatter(msk, gen)
    return out


def specialize_frc(mol: Dict[str, Tensor], msk: Optional[Tensor], gen: Tensor):
    out = mol.copy()
    if msk is None:
        out[p.frc] = gen.view_as(mol[p.pos])
    else:
        out[p.frc] = out[p.pos].masked_scatter(msk, gen).masked_fill(~msk, 0.0)
    return out


def generalize_add_cel(mol: Dict[str, Tensor], gen):
    cel = mol[p.cel].flatten(1)
    return cat(gen, cel)


def specialize_del_cel(mol: Dict[str, Tensor], gen: Tensor):
    num_dim = mol[p.cel].size(2)
    out = mol.copy()
    num_two = num_dim * num_dim
    cel, gen = split(gen, num_two)
    out[p.cel] = cel.view((-1, num_dim, num_dim))
    return out, gen


def specialize_del_sts(mol: Dict[str, Tensor], gen: Tensor):
    num_dim = mol[p.cel].size(2)
    out = mol.copy()
    num_two = num_dim * num_dim
    tmp, gen = split(gen, num_two)
    cel_grd = tmp.view((-1, num_dim, num_dim))
    cel_det = torch.det(out[p.cel])
    out[p.sts] = (cel_grd @ mol[p.cel]) / cel_det[:, None, None]
    return out, gen


def generalize_add_mul(mol: Dict[str, Tensor], gen: Tensor):
    lam = mol[p.con_mul]
    return cat(lam, gen)


def specialize_del_mul(mol: Dict[str, Tensor], gen: Tensor):
    out = mol.copy()
    num_con = mol[p.con_cen].size(1)
    lam, gen = split(gen, num_con)
    out[p.con_mul] = lam
    return out, gen


def specialize_del_mul_frc(mol: Dict[str, Tensor], gen: Tensor):
    out = mol.copy()
    num_con = mol[p.con_cen].size(1)
    lam, gen = split(gen, num_con)
    out[p.con_mul_frc] = lam
    return out, gen
