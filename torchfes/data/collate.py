from typing import List, Union
from ase import Atoms
import numpy as np
import torch
from .ase import atoms_to_dict
from .numpy import stack
from .. import properties as p


def sym_to_elm(symbols: Union[str, List, np.ndarray],
               order: Union[np.ndarray, List[str]]):
    """Transform symbols to elements."""
    if not isinstance(order, list):
        order = order.tolist()
    if not isinstance(symbols, (str, list)):
        symbols = symbols.tolist()
    if isinstance(symbols, str):
        if symbols in order:
            return order.index(symbols)
        else:
            return -1
    else:
        return np.array([sym_to_elm(s, order) for s in symbols])


def collate_single(data):
    if isinstance(data, Atoms):
        return collate_single(atoms_to_dict(data))
    else:
        assert isinstance(data, dict)
        return {key: np.array(val) for key, val in data.items()}


class ToDictArray:
    def __init__(self, default_values=None):
        if default_values is None:
            default_values = {}
        self.default_values = {
            p.pos: 0.0,
            p.sym: '',
            p.elm: -1,
            p.cel: 0.0,
            p.pbc: True,
            p.mas: 1.0,
        }
        for key, val in default_values.items():
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
    def __init__(self, symbols: List[str], default_values=None):
        self.symbols = symbols
        self.to_dict_array = ToDictArray(default_values)

    def __call__(self, datas):
        dict_arrays = self.to_dict_array(datas)
        if p.sym in dict_arrays:
            assert p.elm not in dict_arrays
            dict_arrays[p.elm] = sym_to_elm(
                dict_arrays.pop(p.sym), self.symbols)
        dict_tensors = {key: torch.tensor(val.tolist())
                        for key, val in dict_arrays.items()}
        return dict_tensors
