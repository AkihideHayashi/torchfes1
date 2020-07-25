from typing import List
from ase import Atoms
import numpy as np
import torch
from .convert import _atoms_to_array, sym_to_elm
from .numpy import stack
from .. import properties as p
from .default import default_values


def collate_single(data):
    if isinstance(data, Atoms):
        return collate_single(_atoms_to_array(data))
    else:
        assert isinstance(data, dict), type(data)
        return {key: np.array(val) for key, val in data.items()}


class ToDictArray:
    def __init__(self, default=None):
        if default is None:
            default = {}
        self.default_values = default_values.copy()
        for key, val in default.items():
            self.default_values[key] = val

    def __call__(self, datas):
        list_dict_arrays = [collate_single(data) for data in datas]
        keys = set()
        for da in list_dict_arrays:
            keys.update(da.keys())
        dict_arrays = {
            key: stack(
                [da[key] for da in list_dict_arrays],
                self.default_values[key]
            ) for key in keys
        }
        return dict_arrays


class ToDictTensor:
    def __init__(self, symbols: List[str], default=None):
        self.symbols = symbols
        self.to_dict_array = ToDictArray(default)

    def __call__(self, datas):
        dict_arrays = self.to_dict_array(datas)
        if p.sym in dict_arrays:
            assert p.elm not in dict_arrays
            dict_arrays[p.elm] = sym_to_elm(
                dict_arrays.pop(p.sym), self.symbols)
        if p.ent not in dict_arrays:
            dict_arrays[p.ent] = dict_arrays[p.elm] >= 0
        dict_tensors = {key: torch.tensor(val.tolist())
                        for key, val in dict_arrays.items()}
        return dict_tensors
