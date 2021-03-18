from typing import List
from ase import Atoms
import torch
from torch import Tensor
from .convert import from_atoms
from .mol import cat
from ..mol import add_basic


class ToDictTensor:
    def __init__(self, symbols: List[str], default_precision=True):
        self.symbols = symbols
        self.default_precision = default_precision

    def __call__(self, datas):
        return cat([self._to_dict_tensor(data) for data in datas])

    def _to_dict_tensor(self, data):
        if isinstance(data, Atoms):
            return from_atoms(data, self.symbols)
        else:
            assert isinstance(data, dict)
            for key in data:
                assert isinstance(data[key], Tensor), (key, type(data[key]))
                if self.default_precision:
                    data[key] = torch.tensor(data[key].tolist())
            data = add_basic(data)
            return data
