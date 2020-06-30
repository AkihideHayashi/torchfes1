import numpy as np
from ase import Atoms
import torchfes as fes
from torchfes.recorder import PathPair
from torchfes import properties as p


order = np.array(['H', 'C', 'N', 'O'])
with fes.rec.open_torch(PathPair('trj'), 'r') as f:
    data = f[-1]

elm = data[p.elm].squeeze(0).numpy()
pos = data[p.pos].squeeze(0).numpy()
ent = elm >= 0

atoms = Atoms(order[elm[ent]], pos[ent])
print(atoms)
atoms.write('test.xyz')