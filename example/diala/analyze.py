from ase.units import kB, fs
import matplotlib.pyplot as plt
import torch
from torchfes.recorder.torch import read_trj
from torchfes import properties as p, functional as fn
from torchfes.colvar.dihedral import dihedral


pos, tim, mom, mas, ent = read_trj('trj.pt',
                                   [p.pos, p.tim, p.mom, p.mas, p.ent])

num_dihed = torch.tensor([[11, 9, 15, 17], [11, 9, 7, 5]]) - 1
colvar = torch.stack([dihedral(pos[i], num_dihed) for i in range(pos.size(0))])

kin = fn.kinetic_energies_trajectory(mom, mas, ent)
kbt = fn.temperatures_trajectory(kin, ent, 3, 3)
print(kbt.mean() / kB)

plt.plot(tim, colvar[:, 0, 0])
plt.plot(tim, colvar[:, 0, 1])
plt.show()

plt.plot(tim / fs, kbt / kB)
plt.axhline(300)
plt.show()
