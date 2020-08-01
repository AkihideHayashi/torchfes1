from typing import Dict, Union, List
from ase import Atoms
import numpy as np
import torch
from torch import Tensor
from .mol import Unbind, PakAtm
from ..mol import add_basic
from .. import properties as p

_unbind = Unbind([PakAtm()])


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


def elm_to_sym(elm: np.array, sym: np.array):
    ent = elm >= 0
    return np.where(ent, sym[elm], '')


def to_atoms(mol: Dict[str, Tensor], sym: List[str]):
    assert mol[p.pos].size(0) == 1
    arr = {key: val.clone().detach().squeeze(0).numpy()
           for key, val in mol.items()}
    arr[p.sym] = np.array(sym)[arr.pop(p.elm)]
    return _array_to_atoms(arr)


def to_atoms_list(mol: Dict[str, Tensor], sym: List[str]):
    return [to_atoms(mol, sym) for mol in _unbind(mol)]


def from_atoms(atoms: Atoms, sym: List[str]):
    arr = _atoms_to_array(atoms)
    arr[p.elm] = sym_to_elm(arr.pop(p.sym), sym)
    tmp = {
        key: torch.tensor(val.tolist()).unsqueeze(0)
        for key, val in arr.items()
    }
    ret = add_basic(tmp)
    return ret


def from_atoms_list(atoms_list: List[Atoms], sym: List[str]):
    return [from_atoms(atoms, sym) for atoms in atoms_list]


def _array_to_atoms(atoms: Dict[str, np.ndarray]):
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
    ret = Atoms(symbols=sym, positions=pos, cell=cel, pbc=pbc,
                masses=mas, momenta=mom)
    if p.idt in atoms:
        ret.info[p.idt] = atoms[p.idt]
    return ret


def _atoms_to_array(atoms: Atoms):
    ret = {
        p.pos: atoms.positions,
        p.cel: np.array(atoms.cell),
        p.pbc: atoms.pbc,
        p.mas: atoms.get_masses(),
        p.sym: np.array(atoms.get_chemical_symbols()),
    }
    if p.idt in atoms.info:
        ret[p.idt] = np.array(atoms.info[p.idt])
    return ret
