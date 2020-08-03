from typing import Dict, Optional
from torch import nn, Tensor
from ...colvar.fix import FixGen
from ... import properties as p
from .utils import generalize, specialize_pos, specialize_grd


class Generalize(nn.Module):
    def __init__(self,
                 con: Optional[nn.Module],
                 fix: Optional[FixGen], use_cel: bool = False):
        super().__init__()
        if fix is None:
            self.fix = None
        else:
            self.fix = fix.fix
        if con is None:
            self.con = None
        else:
            self.con = con
        self.use_cel = use_cel

    def forward(self, mol: Dict[str, Tensor]):
        mol = mol.copy()
        if self.fix is None:
            msk = None
        else:
            msk = ~self.fix(mol)
            mol[p.fix_msk] = msk
        mol = generalize(mol, )
        gen = _generalize_pos(mol, msk)

        if self.use_cel:
            gen = _generalize_add_cel(mol, gen)

        if self.con is not None:
            gen = _generalize_add_mul(mol, gen)

        gen = gen.clone().detach().requires_grad_()
        mol[p.gen_pos] = gen

        if self.con is not None:
            mol, gen = _specialize_del_mul(mol, gen)
        if self.use_cel:
            mol, gen = _specialize_del_cel(mol, gen)
        mol = _specialize_pos(mol, msk, gen)
        return mol


class Specialize(nn.Module):
    def __init__(self,
                 con: Optional[nn.Module],
                 fix: Optional[FixGen], use_cel: bool = False):
        super().__init__()
        if fix is None:
            self.fix = None
        else:
            self.fix = fix.fix
        if con is None:
            self.con = None
        else:
            self.con = con
        self.use_cel = use_cel

    def forward(self, mol: Dict[str, Tensor]):
        if self.fix is None:
            msk = None
        else:
            msk = ~self.fix(mol)

        gen = mol[p.gen_pos]
        if self.con is not None:
            mol, gen = _specialize_del_mul(mol, gen)
        if self.use_cel:
            mol, gen = _specialize_del_cel(mol, gen)
        mol = _specialize_pos(mol, msk, gen)

        gen = - mol[p.gen_grd]
        if self.con is not None:
            mol, gen = _specialize_del_mul_frc(mol, gen)
        if self.use_cel:
            mol, gen = _specialize_del_sts(mol, gen)
        mol = _specialize_frc(mol, msk, gen)
        return mol
