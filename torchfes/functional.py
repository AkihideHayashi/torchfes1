from typing import Dict, Union

import torch
from torch import Tensor

from . import properties as p


def maxwell(pos: Tensor, ent: Tensor, mas: Tensor, kbt: Tensor):
    """
    N ~ exp(-x^2/2s)
    v_i ~ exp(-mv^2/2kT)
    p_i ~ exp(-p^2/2mkT)
    s = mkT
    """
    randn = torch.randn_like(pos)
    mom = randn * (mas[:, :, None] * kbt[:, None, None]).sqrt()
    return mom * ent[:, :, None]


def kinetic_energies(mom: Tensor, mas: Tensor, ent: Tensor):
    kin = (0.5 * mom * mom / mas[:, :, None]).sum(-1)
    return (kin * ent).sum(-1)


def kinetic_energies_trajectory(mom: Tensor, mas: Tensor, ent: Tensor):
    return torch.stack([
        kinetic_energies(mom_i, mas_i, ent_i) for mom_i, mas_i, ent_i in
        zip(mom, mas, ent)])


def temperatures_trajectory(kin: Tensor, ent: Tensor,
                            ddof: Union[int, Tensor] = 3, ndim: int = 3):
    return torch.stack([
        temperatures(kin_i, ent_i, ddof, ndim) for kin_i, ent_i in
        zip(kin, ent)])


def temperatures(kin: Tensor, ent: Tensor,
                 ddof: Union[Tensor, int] = 3, ndim: int = 3):
    if ddof is int:
        ddof = torch.ones_like(kin) * ddof
    dof = (ent.sum(-1) * ndim) - ddof
    return kin / dof * 2


def noise(frc: Tensor, ent: Tensor) -> Tensor:
    return torch.randn_like(frc) * ent


def nhc_conserve(inp: Dict[str, Tensor], ddof: Tensor):
    out = inp.copy()
    mas = out[p.mas][:, :, None]
    ent = out[p.ent][:, :, None]
    _, n_atm, n_dim = inp[p.pos].size()
    dof = torch.ones_like(ddof) * n_atm * n_dim - ddof
    term1 = (0.5 * inp[p.mom] ** 2 / mas * ent).sum(-1).sum(-1)
    term2 = (0.5 * inp[p.mom_nhc] ** 2 / inp[p.mas_nhc]).sum(-1)
    term3 = (inp[p.kbt][:, None] * inp[p.pos_nhc])
    term3[:, 0] *= dof
    term3 = term3.sum(-1)
    out[p.con_nhc] = term1 + term2 + term3 + inp[p.eng_mol]
    return out


def max_frc(inp: Dict[str, Tensor], n_elm: int):
    elm = inp[p.elm]
    frc = inp[p.frc].norm(p=2, dim=-1)
    forces = []
    for i in range(n_elm):
        frc_elm = torch.masked_fill(frc, elm == i, 0.0)
        frc_elm_max, _ = frc_elm.max(dim=1)
        forces.append(frc_elm_max)
    return torch.stack(forces, dim=1)
