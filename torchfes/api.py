from typing import NamedTuple
from torch import Tensor


class Energies(NamedTuple):
    eng_mol: Tensor
    eng_atm: Tensor
    eng_mol_std: Tensor
    eng_atm_std: Tensor
