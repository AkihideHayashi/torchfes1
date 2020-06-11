from ase import Atoms
import numpy as np
from .. import properties as p


def atoms_to_dict(atoms: Atoms):
    return {
        p.pos: atoms.positions,
        p.cel: np.array(atoms.cell),
        p.pbc: atoms.pbc,
        p.mas: atoms.get_masses(),
        p.sym: np.array(atoms.get_chemical_symbols()),
    }
