from .coordination import Rational
from math import inf
from typing import List, Dict
import torch
from torch import Tensor, nn
from .. import properties as p
from pointneighbor import functional as fn, PntFul, pnt_ful


def center_of_mass(pos: Tensor, mas: Tensor, msk: Tensor):
    pos_mas = pos * mas[:, :, None]
    num = pos_mas.masked_fill(~msk[:, :, None], 0.0).sum(dim=1, keepdim=True)
    den = mas.masked_fill(~msk, 0.0).sum(dim=1, keepdim=True)[:, :, None]
    return num / den


def element_mask(mol: Dict[str, Tensor], key: Tensor):
    elm = mol[p.elm]
    return (elm[:, :, None] == key[None, None, :]).any(dim=2)


def to_unit_cell(pos: Tensor, cel_mat: Tensor, cel_inv: Tensor, pbc: Tensor):
    pos_cel = pos @ cel_inv
    spc_cel = fn.get_pos_spc(pos_cel, pbc)
    spc_xyz = spc_cel @ cel_mat
    return fn.to_unit_cell(pos, spc_xyz)


def mol_slb_distances(pe: PntFul, mas: Tensor, max_rpt: Tensor,
                      msk_mol: Tensor, msk_slb: Tensor, a: float):
    sft_cel = fn.arange_prod(max_rpt * 2 + 1) - max_rpt
    sft_xyz = sft_cel.to(pe.cel_mat) @ pe.cel_mat.detach()
    _, n_pnt, _ = pe.pos_xyz.size()
    pos_mol = to_unit_cell(center_of_mass(
        pe.pos_xyz, mas, msk_mol), pe.cel_mat, pe.cel_inv, pe.pbc)
    pos_all = fn.to_unit_cell(pe.pos_xyz, pe.spc_xyz)
    pos_all.masked_fill_(msk_mol[:, :, None], 0.0)
    pos_xyz_i = pos_mol[:, :, None, None, :]
    pos_xyz_j = pos_all[:, None, :, None, :]
    sft_xyz_ij = sft_xyz[:, None, None, :, :]
    vec = fn.vector(pos_xyz_i, pos_xyz_j, sft_xyz_ij)
    sod = fn.square_of_distance(vec, dim=-1)
    dis = sod.sqrt()
    return softmin(dis, msk_slb, a)


def softmin(dis: Tensor, msk: Tensor, a):
    dis = dis.masked_fill(~msk[:, None, :, None], inf)
    dis_min, _ = dis.detach().flatten(2).min(dim=2)
    dd = dis - dis_min[:, :, None, None]
    w = torch.exp(- a * dd)
    dd_ = dd.masked_fill(~msk[:, None, :, None], 0.0)
    w_ = w.masked_fill(~msk[:, None, :, None], 0.0)
    num = (dd_ * w_).flatten(2).sum(dim=2)
    den = w.flatten(2).sum(dim=2)
    return num / den + dis_min


class SurfaceSeparationDistanceElmCom(nn.Module):
    """Softmin of distance"""
    pbc: Tensor
    mol: Tensor
    slb: Tensor
    max_rpt: Tensor

    def __init__(self, mol: List[int], slb: List[int], wdt: float,
                 max_rpt: List[int]):
        super().__init__()
        self.a = 0.5 / (wdt * wdt)
        self.register_buffer('slb', torch.tensor(slb))
        self.register_buffer('mol', torch.tensor(mol))
        self.register_buffer('pbc', torch.full([1], inf))
        self.register_buffer('max_rpt', torch.tensor(max_rpt))

    def forward(self, mol: Dict[str, Tensor]):
        msk_slb = element_mask(mol, self.slb)
        msk_mol = element_mask(mol, self.mol)
        pe = pnt_ful(mol[p.cel], mol[p.pbc], mol[p.pos], mol[p.ent])
        return mol_slb_distances(pe, mol[p.mas], self.max_rpt,
                                 msk_mol, msk_slb, self.a)


class SurfaceSeparationDistanceIdx(nn.Module):
    pbc: Tensor
    slb: Tensor
    max_rpt: Tensor

    def __init__(self, atm: int, slb: List[int], wdt: float,
                 max_rpt: List[int]):
        super().__init__()
        self.a = 0.5 / (wdt * wdt)
        self.register_buffer('slb', torch.tensor(slb))
        self.atm = atm
        self.register_buffer('pbc', torch.full([1], inf))
        self.register_buffer('max_rpt', torch.tensor(max_rpt))

    def forward(self, mol: Dict[str, Tensor]):
        msk_slb = element_mask(mol, self.slb)
        msk_slb[:, self.atm] = False
        n = msk_slb.size(1)
        atm = torch.arange(n, device=msk_slb.device) == self.atm % n
        msk_mol = atm[None, :].expand_as(mol[p.elm])
        pe = pnt_ful(mol[p.cel], mol[p.pbc], mol[p.pos], mol[p.ent])
        return mol_slb_distances(pe, mol[p.mas], self.max_rpt,
                                 msk_mol, msk_slb, self.a)


def softmin_val(val: Tensor, dis: Tensor, msk: Tensor, a):
    dis = dis.masked_fill(~msk[:, None, :, None], inf)
    dis_min, _ = dis.detach().flatten(2).min(dim=2)
    dd = dis - dis_min[:, :, None, None]
    w = torch.exp(- a * dd)
    w_ = w.masked_fill(~msk[:, None, :, None], 0.0)
    vv_ = val.masked_fill_(~msk[:, None, :, None], 0.0)
    num = (vv_ * w_).flatten(2).sum(dim=2)
    den = w.flatten(2).sum(dim=2)
    return num / den


def mol_slb_cos(pe: PntFul, mas: Tensor, max_rpt: Tensor,
                msk_mol: Tensor, msk_slb: Tensor, a: float, dim: int):
    sft_cel = fn.arange_prod(max_rpt * 2 + 1) - max_rpt
    sft_xyz = sft_cel.to(pe.cel_mat) @ pe.cel_mat.detach()
    _, n_pnt, _ = pe.pos_xyz.size()
    pos_mol = to_unit_cell(center_of_mass(
        pe.pos_xyz, mas, msk_mol), pe.cel_mat, pe.cel_inv, pe.pbc)
    pos_all = fn.to_unit_cell(pe.pos_xyz, pe.spc_xyz)
    pos_all.masked_fill_(msk_mol[:, :, None], 0.0)
    pos_xyz_i = pos_mol[:, :, None, None, :]
    pos_xyz_j = pos_all[:, None, :, None, :]
    sft_xyz_ij = sft_xyz[:, None, None, :, :]
    vec = fn.vector(pos_xyz_i, pos_xyz_j, sft_xyz_ij)
    sod = fn.square_of_distance(vec, dim=-1)
    dis = sod.sqrt()
    cos = vec[:, :, :, :, dim] / dis
    return softmin_val(cos, dis, msk_slb, a)


class SurfaceSeparationCosIdx(nn.Module):
    pbc: Tensor
    slb: Tensor
    max_rpt: Tensor

    def __init__(self, atm: int, slb: List[int], wdt: float,
                 max_rpt: List[int], dim: int = 2):
        super().__init__()
        self.a = 0.5 / (wdt * wdt)
        self.register_buffer('slb', torch.tensor(slb))
        self.atm = atm
        self.register_buffer('pbc', torch.full([1], inf))
        self.register_buffer('max_rpt', torch.tensor(max_rpt))
        self.dim = dim

    def forward(self, mol: Dict[str, Tensor]):
        msk_slb = element_mask(mol, self.slb)
        msk_slb[:, self.atm] = False
        n = msk_slb.size(1)
        atm = torch.arange(n, device=msk_slb.device) == self.atm % n
        msk_mol = atm[None, :].expand_as(mol[p.elm])
        pe = pnt_ful(mol[p.cel], mol[p.pbc], mol[p.pos], mol[p.ent])
        return mol_slb_cos(pe, mol[p.mas], self.max_rpt,
                           msk_mol, msk_slb, self.a, self.dim)


def distances_idx(mol: Dict[str, Tensor], i: Tensor, j: Tensor, max_rpt: Tensor):
    pe = pnt_ful(mol[p.cel], mol[p.pbc], mol[p.pos], mol[p.ent])
    sft_cel = fn.arange_prod(max_rpt * 2 + 1) - max_rpt
    sft_xyz = sft_cel.to(pe.cel_mat) @ pe.cel_mat.detach()
    pos = fn.to_unit_cell(pe.pos_xyz, pe.spc_xyz)
    pos_i = pos[:, i, :]
    pos_j = pos[:, j, :]
    vec = fn.vector(
        pos_i[:, :, None, None, :],
        pos_j[:, None, :, None, :],
        sft_xyz[:, None, None, :, :]
    )
    dis = vec.pow(2).sum(dim=-1).sqrt()
    return dis


def soft_nth(dis, n, sigma):
    dis = torch.sort(dis, dim=1)[0]
    dis_n = dis[:, n]
    dif = dis[:, :, None] - dis_n[:, None, :]
    coef = torch.exp(-dif * dif * 0.5 / (sigma * sigma))
    return (dis[:, :, None] * coef).sum(1) / coef.sum(1)


def to_tensor(x):
    if isinstance(x, list):
        return torch.tensor(x)
    elif isinstance(x, Tensor):
        return x
    elif isinstance(x, int):
        return torch.tensor([x])
    else:
        raise KeyError()


class NthDistance(nn.Module):
    i: Tensor
    j: Tensor
    n: Tensor
    max_rpt: Tensor

    def __init__(self, i, j, n, max_rpt, sigma: float) -> None:
        super().__init__()
        self.register_buffer('i', to_tensor(i))
        self.register_buffer('j', to_tensor(j))
        self.register_buffer('n', to_tensor(n))
        self.register_buffer('max_rpt', to_tensor(max_rpt))
        self.sigma = sigma

    def forward(self, mol: Dict[str, Tensor]):
        dis = distances_idx(mol, self.i, self.j, self.max_rpt)
        dis = dis.flatten(1)
        return soft_nth(dis, self.n, self.sigma)
