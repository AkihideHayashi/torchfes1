from typing import Dict, NamedTuple

from torch import Tensor, nn

import pointneighbor as pn

from . import properties as p


class Energies(NamedTuple):
    eng_mol: Tensor
    eng_atm: Tensor
    eng_mol_std: Tensor
    eng_atm_std: Tensor


def pnt_ful(inp: Dict[str, Tensor]):
    return pn.pnt_ful(cel=inp[p.cel], pbc=inp[p.pbc],
                      pos=inp[p.pos], ent=inp[p.ent])


class Unit(nn.Module):
    def __init__(self, mdl, u):
        super().__init__()
        self.mdl = mdl
        self.u = u

    def forward(self, inp: Dict[str, Tensor], adj: pn.AdjSftSpc):
        eng: Energies = self.mdl(inp, adj)
        return Energies(
            eng_mol=eng.eng_mol * self.u,
            eng_atm=eng.eng_atm * self.u,
            eng_mol_std=eng.eng_mol_std * self.u,
            eng_atm_std=eng.eng_atm_std * self.u
        )
