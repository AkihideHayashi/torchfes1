from typing import Dict, NamedTuple

from torch import Tensor

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
