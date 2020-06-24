from typing import Dict, NamedTuple, Optional

import torch
from torch import Tensor, nn


class Energies(NamedTuple):
    eng_mol: Tensor
    eng_atm: Tensor
    eng_mol_std: Tensor
    eng_atm_std: Tensor


def energies(eng_mol: Tensor, eng_atm: Tensor,
             eng_mol_std: Optional[Tensor] = None,
             eng_atm_std: Optional[Tensor] = None):
    if eng_mol_std is None:
        eng_mol_std = torch.zeros_like(eng_mol)
    if eng_atm_std is None:
        eng_atm_std = torch.zeros_like(eng_atm)
    return Energies(eng_mol=eng_mol, eng_atm=eng_atm,
                    eng_mol_std=eng_mol_std, eng_atm_std=eng_atm_std)


class Unit(nn.Module):
    def __init__(self, mdl, u):
        super().__init__()
        self.mdl = mdl
        self.u = u

    def forward(self, inp: Dict[str, Tensor]):
        eng: Energies = self.mdl(inp)
        return Energies(
            eng_mol=eng.eng_mol * self.u,
            eng_atm=eng.eng_atm * self.u,
            eng_mol_std=eng.eng_mol_std * self.u,
            eng_atm_std=eng.eng_atm_std * self.u
        )
