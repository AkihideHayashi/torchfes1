from typing import Dict, Union, List
from ase import Atoms
import numpy as np
import torch
from torch import Tensor
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


def elm_to_sym(elm: np.array, order: np.array):
    ent = elm >= 0
    return np.where(ent, order[elm], '')


def dict_array_to_single_dict_array_list(
        inp: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    n = inp[p.pos].shape[0]
    for val in inp.values():
        assert val.shape[0] == n, val.shape
    return [{key: val[i] for key, val in inp.items()} for i in range(n)]


def atoms_to_single_dict_array(atoms: Atoms):
    return {
        p.pos: atoms.positions,
        p.cel: np.array(atoms.cell),
        p.pbc: atoms.pbc,
        p.mas: atoms.get_masses(),
        p.sym: np.array(atoms.get_chemical_symbols()),
    }


def single_dict_array_to_atoms(atoms: Dict[str, np.ndarray]):
    assert atoms[p.sym].ndim == 1
    ent = atoms[p.sym] != ''
    sym = atoms[p.sym][ent]
    cel = atoms[p.cel]
    pbc = atoms[p.pbc]
    pos = atoms[p.pos][ent]
    mas = None
    if p.mas in atoms:
        mas = atoms[p.mas][ent]
    mom = None
    if p.mom in atoms:
        mom = atoms[p.mom][ent]
    return Atoms(symbols=sym, positions=pos, cell=cel, pbc=pbc,
                 masses=mas, momenta=mom)


def dict_array_to_dict_tensor(atoms: Dict[str, np.ndarray], order: List[str]):
    atoms = atoms.copy()
    atoms[p.elm] = sym_to_elm(atoms.pop(p.sym), order)
    return {key: torch.tensor(val.tolist()) for key, val in atoms.items()}


def dict_tensor_to_dict_array(atoms: Dict[str, Tensor], order: List[str]):
    symbols = np.array(order)
    atoms = {key: val.numpy() for key, val in atoms.items()}
    atoms[p.sym] = elm_to_sym(atoms.pop(p.elm), symbols)
    return atoms


def dict_array_to_atoms_list(atoms: Dict[str, np.ndarray]):
    return [single_dict_array_to_atoms(atm) for atm
            in dict_array_to_single_dict_array_list(atoms)]


def dict_tensor_to_atoms_list(atoms: Dict[str, Tensor], order: List[str]):
    return [
        single_dict_array_to_atoms(atm) for atm in
        dict_array_to_single_dict_array_list(
            dict_tensor_to_dict_array(atoms, order))
    ]
