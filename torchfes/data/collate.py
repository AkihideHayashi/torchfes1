from typing import List
from ase import Atoms
from torch import Tensor
from .convert import from_atoms
from .mol import stack


class ToDictTensor:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols

    def __call__(self, datas):
        return stack([self._to_dict_tensor(data) for data in datas])

    def _to_dict_tensor(self, data):
        if isinstance(data, Atoms):
            return from_atoms(data, self.symbols)
        else:
            assert isinstance(data, dict)
            for key in data:
                assert isinstance(data[key], Tensor)
            return data
