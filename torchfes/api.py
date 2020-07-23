from typing import Dict, NamedTuple, Optional

import torch
from torch import Tensor, nn


class Energies(NamedTuple):
    eng_mol: Tensor
    eng_atm: Tensor
    eng_mol_std: Tensor
    eng_atm_std: Tensor
    eng_mol_ens: Tensor
    eng_atm_ens: Tensor


def ensemble_energies_from_atomic(eng_atm_ens: Tensor):
    assert eng_atm_ens.dim() == 3  # bch, ens, atm
    eng_mol_ens = eng_atm_ens.sum(dim=2)
    eng_mol = eng_mol_ens.mean(dim=1)
    eng_atm = eng_atm_ens.mean(dim=1)
    eng_mol_std = eng_mol_ens.std(dim=1, unbiased=True)
    eng_atm_std = eng_atm_ens.std(dim=1, unbiased=True)
    return Energies(
        eng_mol=eng_mol, eng_atm=eng_atm,
        eng_mol_std=eng_mol_std, eng_atm_std=eng_atm_std,
        eng_mol_ens=eng_mol_ens, eng_atm_ens=eng_atm_ens,
    )


def energies(eng_mol: Tensor, eng_atm: Tensor,
             eng_mol_std: Optional[Tensor] = None,
             eng_atm_std: Optional[Tensor] = None):
    import warnings
    warnings.warn('energies is deprecated.')
    if eng_mol_std is None:
        eng_mol_std = torch.zeros_like(eng_mol)
    if eng_atm_std is None:
        eng_atm_std = torch.zeros_like(eng_atm)
    return Energies(eng_mol=eng_mol, eng_atm=eng_atm,
                    eng_mol_std=eng_mol_std, eng_atm_std=eng_atm_std,
                    eng_mol_ens=eng_mol[None, :, :],
                    eng_atm_ens=eng_atm[None, :, :]
                    )


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
            eng_atm_std=eng.eng_atm_std * self.u,
            eng_mol_ens=eng.eng_mol_ens * self.u,
            eng_atm_ens=eng.eng_atm_ens * self.u,
        )
