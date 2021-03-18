from typing import Optional, Dict

from torch import nn, Tensor

from ..fes.bme import BMEJac, BMEShk, BMERtl, BMEDetLmd
from .unified import UpdtTim, UpdtPos, UpdtMom, UpdtKbt
from ..forcefield import EvlAdjEngFrc
from .rpmd import RpmdBase, RpmdKin, RpmdLangevin, RpmdPos


class Sequential(nn.Sequential):
    def forward(self, input: Dict[str, Tensor]):
        for module in self:
            input = module(input)
        return input


class MolecularDynamics:
    def __init__(self, dyn, callback):
        self.dyn = dyn
        self.callback = callback

    def _step(self, mol):
        mol = self.dyn(mol)
        self.callback(mol)
        return mol

    def __call__(self, mol, n=1):
        for i in range(n):
            mol = self._step(mol)
        return mol


class MetaDynamics:
    def __init__(self, dyn, mtd, n_mtd, callback):
        self.dyn = dyn
        self.mtd = mtd
        self.n_mtd = n_mtd
        self.callback = callback

    def _step(self, mol):
        mol = self.dyn(mol)
        self.callback(mol)
        return mol

    def __call__(self, mol, n=1):
        for i in range(n):
            mol = self._step(mol)
            if i % self.n_mtd == self.n_mtd - 1:
                mol = self.mtd(mol)
        return mol


def leap_frog(evl: EvlAdjEngFrc,
              kbt: Optional[nn.Module] = None,
              con: Optional[nn.Module] = None, tol: float = 1e-5,
              bme: bool = True):
    tim = UpdtTim(1.0)
    pos = UpdtPos(1.0)
    if kbt is None:
        mom = UpdtMom(1.0)
    else:
        kbt = UpdtKbt(kbt, 1.0)
        mom = UpdtMom(0.5)

    if con is None:
        if kbt is None:
            return Sequential(mom, pos, tim, evl)
        else:
            return Sequential(mom, kbt, mom, pos, tim, evl)
    else:
        shk = BMEShk(con, tol)
        det = BMEDetLmd()
        jac = BMEJac(con, bme)
        if kbt is None:
            return Sequential(mom, pos, shk, tim, det, evl, jac)
        else:
            return Sequential(mom, kbt, mom, pos, shk, tim, det, evl, jac)


def velocity_verlet(evl: EvlAdjEngFrc,
                    kbt: Optional[nn.Module] = None,
                    con: Optional[nn.Module] = None, tol: float = 1e-5,
                    bme: bool = True):
    tim = UpdtTim(1.0)
    pos = UpdtPos(1.0)
    mom = UpdtMom(0.5)
    if kbt is not None:
        kbt = UpdtKbt(kbt, 0.5)
    if con is None:
        if kbt is None:
            return nn.Sequential(mom, pos, tim, evl, mom)
        else:
            return nn.Sequential(kbt, mom, pos, tim, evl, mom, kbt)
    else:
        shk = BMEShk(con, tol)
        rtl = BMERtl(con, tol)
        det = BMEDetLmd()
        jac = BMEJac(con, bme)
        if kbt is None:
            return nn.Sequential(mom, pos, shk, tim, det, evl, jac, mom, rtl)
        else:
            return nn.Sequential(
                kbt, mom, pos, shk, tim, det, evl, jac, mom, kbt, rtl)


# def ring_polymer_velocity_verlet(evl: EvlAdjEngFrc,
#                                  kbt: Optional[nn.Module] = None,
#                                  con: Optional[nn.Module] = None,
#                                  tol: float = 1e-5,
#                                  bme: bool = True):
#     tim = UpdtTim(1.0)
#     base = 