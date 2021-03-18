from typing import Callable, Optional, Dict

import torch
from torch import nn, Tensor

from ..fes.bme import BMEJac, BMEShk, BMERtl, BMEDetLmd
from .unified import UpdtTim, UpdtPos, UpdtMom, UpdtKbt
from ..forcefield import EvlAdjEngFrc
from ..fes.plumed import Plumed
from .rpmd import RpmdPos, RpmdLangevin, RpmdKin, RpmdBase, rpmd_base
from .. import properties as p


def velocity_verlet(mol: Dict[str, Tensor], evl: EvlAdjEngFrc,
                    kbt: nn.Module, callback: Callable, n: int):
    tim = UpdtTim(1.0)
    pos = UpdtPos(1.0)
    kbt = UpdtKbt(kbt, 0.5)
    mom = UpdtMom(0.5)
    mol = evl(mol)
    for i in range(n):
        mol = tim(mol)
        mol = kbt(mol)
        mol = mom(mol)
        mol = pos(mol)
        mol = evl(mol)
        mol = mom(mol)
        mol = kbt(mol)
        callback(i, mol)
    return mol


def velocity_verlet_plumed(mol: Dict[str, Tensor], evl: EvlAdjEngFrc,
                           kbt: nn.Module, plumed: Plumed,
                           callback: Callable, n: int):
    tim = UpdtTim(1.0)
    pos = UpdtPos(1.0)
    kbt = UpdtKbt(kbt, 0.5)
    mom = UpdtMom(0.5)
    mol = evl(mol)
    mol = tim(mol)
    mol = kbt(mol)
    mol = mom(mol)
    mol = pos(mol)
    mol = evl(mol)

    def _apply(mol):
        mol = mom(mol)
        mol = kbt(mol)
        callback(i, mol)
        mol = tim(mol)
        mol = kbt(mol)
        mol = mom(mol)
        mol = pos(mol)
        mol = evl(mol)
        return mol
    for i in range(n):
        mol = plumed(mol)
        mol_bak = mol.copy()
        flag = True
        while flag:
            flag = False
            mol = _apply(mol)
            if _has_nan(mol):
                print('nan rollback', flush=True)
                flag = True
                mol = mol_bak
    return mol


def _has_nan(mol):
    for key in mol:
        if torch.isnan(mol[key]).any():
            return True
    return False


def apply_dyn(dyn, mol, file):
    mol_back = mol.copy()
    flag = True
    while flag:
        flag = False
        mol = dyn(mol)
        if _has_nan(mol):
            print('nan rollback.', file=file)
            flag = True
            mol = mol_back
    return mol


def same(x: Tensor):
    v = x.mean().item()
    assert torch.all((x - v).abs() < 1e-8)
    return v


def ring_polymer_velocity_verlet(hbar: float, tau: float,
                                 mol: Dict[str, Tensor],
                                 evl: EvlAdjEngFrc,
                                 callback: Callable, nstp: int, device):

    for key in mol:
        mol[key] = mol[key].to(device)
    evl = evl.to(device)
    beta = 1 / same(mol[p.kbt])
    n = mol[p.kbt].numel()
    mas = mol[p.mas]
    dtm = same(mol[p.dtm])
    num_dim = mol[p.pos].size(2)
    tim = UpdtTim(1.0)
    base = rpmd_base(hbar, beta, n, device)
    pos = RpmdPos(base, mas, dtm, num_dim).to(device)
    kbt = RpmdLangevin(base, tau, dtm, mas, num_dim).to(device)
    kin = RpmdKin(base).to(device)
    mom = UpdtMom(0.5).to(device)
    mol = evl(mol)
    for i in range(nstp):
        mol = tim(mol)
        mol = kbt(mol)
        mol = mom(mol)
        mol = pos(mol)
        mol = evl(mol)
        mol = kin(mol)
        mol = mom(mol)
        mol = kbt(mol)
        callback(i, mol)
    return mol


def leap_frog_bme(mol: Dict[str, Tensor], evl: EvlAdjEngFrc,
                  kbt: nn.Module, col: nn.Module, tol: float,
                  callback: Callable, n: int, device):
    tim = UpdtTim(1.0).to(device)
    pos = UpdtPos(1.0).to(device)
    kbt = UpdtKbt(kbt, 1.0).to(device)
    mom = UpdtMom(0.5).to(device)

    shk = BMEShk(col, tol).to(device)
    det = BMEDetLmd().to(device)
    jac = BMEJac(col, True).to(device)

    evl = evl.to(device)
    for key in mol:
        mol[key] = mol[key].to(device)

    mol = evl(mol)
    for i in range(n):
        mol = tim(mol)
        mol = mom(mol)
        mol = kbt(mol)
        mol = mom(mol)
        mol = pos(mol)
        mol = shk(mol)
        mol = det(mol)
        mol = evl(mol)
        mol = jac(mol)
        callback(i, mol)
    return mol
