from typing import Union, List
import numpy as np
import torch
from torch import Tensor
from ase import Atoms
from . import properties as p


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


# TODO: add mas
def _ase_to_inp_inner(atoms: Atoms, order: List[str]):
    pos = np.array(atoms.positions).tolist()
    cel = np.array(atoms.cell).tolist()
    sym = atoms.get_chemical_symbols()
    pbc = np.array([True, True, True])
    elm = sym_to_elm(sym, order)
    return {
        p.pos: torch.tensor(pos),
        p.cel: torch.tensor(cel),
        p.pbc: torch.tensor(pbc),
        p.elm: torch.tensor(elm)
    }


def ase_to_inp(atoms_list: List[Atoms], order: List[str]):
    inp_lst = [_ase_to_inp_inner(atoms, order) for atoms in atoms_list]
    pos = pad_cat_torch([inp[p.pos][None, :, :] for inp in inp_lst], 0.0)
    cel = torch.stack([inp[p.cel] for inp in inp_lst])
    elm = pad_cat_torch([inp[p.elm][None, :] for inp in inp_lst], -1)
    pbc = torch.stack([inp[p.pbc] for inp in inp_lst])
    return {p.pos: pos, p.cel: cel, p.elm: elm, p.pbc: pbc}


def pad_torch(tensor: Tensor, size: List[int], value: float):
    pad_: List[int] = []
    for i, s in enumerate(size):
        dif = max(0, s - tensor.size(i))
        pad_.append(dif)
        pad_.append(0)
    pad_.reverse()
    return torch.nn.functional.pad(tensor, pad_,
                                   mode='constant', value=value)


def pad_cat_torch(tensors: List[Tensor], value: float, dim: int = 0):
    t0 = tensors[0]
    for tensor in tensors:
        assert t0.dim() == tensor.dim()
    size: List[int] = []
    for i in range(t0.dim()):
        size.append(max([tensor.size(i) for tensor in tensors]))
    size[dim] = 0
    padded = [pad_torch(tensor, size, value) for tensor in tensors]
    return torch.cat(padded, dim=dim)


def pad_numpy(array: np.ndarray, shape: List[int],
              constant_values: Union[bool, int, float, str]):
    assert array.ndim == len(shape)
    pad_width = [(0, max(0, si - asi)) for si, asi in zip(shape, array.shape)]
    return np.pad(array, pad_width, mode='constant',
                  constant_values=constant_values)


def pad_cat_numpy(arrays: List[np.ndarray],
                  constant_values: Union[bool, int, float, str], axis=0):
    for arr in arrays:
        assert arr.ndim == arrays[0].ndim
    shape = []
    for i in range(arrays[0].ndim):
        shape.append(max(arr.shape[i] for arr in arrays))
    shape[axis] = 0
    padded = [pad_numpy(arr, shape, constant_values) for arr in arrays]
    return np.concatenate(padded, axis=axis)


def pad(tensor, size, value):
    if isinstance(tensor, Tensor):
        return pad_torch(tensor, size, value)
    elif isinstance(tensor, np.ndarray):
        return pad_numpy(tensor, size, value)
    else:
        raise RuntimeError()


def pad_cat(tensors, value, dim=0):
    t0 = tensors[0]
    if isinstance(t0, Tensor):
        return pad_cat_torch(tensors, value, dim)
    elif isinstance(t0, np.ndarray):
        return pad_cat_numpy(tensors, value, dim)
    else:
        raise RuntimeError()
