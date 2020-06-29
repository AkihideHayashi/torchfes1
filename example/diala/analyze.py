from math import pi
from typing import Dict
from pathlib import Path
from ase.units import kB, fs
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torchfes.recorder import open_torch_mp
from torchfes import properties as p, functional as fn
from torchfes.colvar.dihedral import dihedral


class ColVar(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, inp: Dict[str, Tensor]):
        mom = inp[p.mom]
        mas = inp[p.mas]
        ent = inp[p.ent]
        kin = fn.kinetic_energies(mom, mas, ent)
        kbt = fn.temperatures(kin, ent, 3)
        return {
            p.tim: inp[p.tim],
            'colvar': dihedral(self.idx, inp[p.pos]),
            p.kbt: kbt
        }


def read_trj(path, proc):
    with open_torch_mp(path, 'r') as f:
        for i, data in enumerate(f):
            print(i)
            yield proc(data)


def make_tmp():
    idx = torch.tensor([[5, 7, 9, 15], [7, 9, 15, 17]]).t() - 1
    col_var = ColVar(idx)
    data = list(read_trj('trj.bk.pt', col_var))
    with open('tmp.pt', 'wb') as f:
        torch.save(data, f)


def main():
    if not Path('tmp.pt').is_file():
        make_tmp()
    data = torch.load('tmp.pt')
    plot(data)


def fes(data):
    tim = torch.stack([d[p.tim] for d in data])
    col = torch.stack([d['colvar'] for d in data])
    col = col[:, 0, 0]
    bins = 100
    x = torch.linspace(-pi, pi, bins)
    y = torch.histc(col, bins=bins, min=-pi, max=pi).log()

    plt.plot(x, y)
    plt.show()


def plot(data):
    tim = torch.stack([d[p.tim] for d in data])
    col = torch.stack([d['colvar'] for d in data])
    kbt = torch.stack([d[p.kbt] for d in data])
    plt.plot(tim[:, 0] / 1000000, col[:, 0, 0], '.')
    plt.plot(tim[:, 0] / 1000000, col[:, 0, 1], '.')
    # plt.plot(tim[:, 0], kbt[:, 0] / kB)
    plt.show()


if __name__ == '__main__':
    main()

# pos, tim, mom, mas, ent = read_trj('trj.pt',
#                                    [p.pos, p.tim, p.mom, p.mas, p.ent])
# num_dihed = torch.tensor([[, 9, 15, 17], [11, 9, 7, 5]]) - 1
# colvar = torch.stack([dihedral(pos[i], num_dihed) for i in range(pos.size(0))])

# kin = fn.kinetic_energies_trajectory(mom, mas, ent)
# kbt = fn.temperatures_trajectory(kin, ent, 3, 3)
# print(kbt.mean() / kB)

# plt.plot(tim, colvar[:, 0, 0])
# plt.plot(tim, colvar[:, 0, 1])
# plt.show()

# plt.plot(tim / fs, kbt / kB)
# plt.axhline(300)
# plt.show()
