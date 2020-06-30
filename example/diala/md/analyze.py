import argparse
from math import pi
from typing import Dict
from pathlib import Path
from ase.units import fs, kB
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torchfes.recorder import open_torch, PathPair
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
        kbt_ = fn.temperatures(kin, ent, 3)
        return {
            p.tim: inp[p.tim],
            'colvar': dihedral(self.idx, inp[p.pos]),
            p.kbt: kbt_
        }


def read_trj(path, proc):
    with open_torch(path, 'r') as f:
        for _, data in enumerate(f):
            yield proc(data)


def make_tmp():
    idx = torch.tensor([[5, 7, 9, 15], [7, 9, 15, 17]]).t() - 1
    col_var = ColVar(idx)
    data = list(read_trj(PathPair('md'), col_var))
    with open('tmp.pt', 'wb') as f:
        torch.save(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exec', choices=['fes', 'col', 'kbt'])
    args = parser.parse_args()
    if not Path('tmp.pt').is_file():
        make_tmp()
    data = torch.load('tmp.pt')
    if args.exec == 'fes':
        fes(data)
    elif args.exec == 'col':
        col(data)
    else:
        assert args.exec == 'kbt'
        kbt(data)


def fes(data):
    colvar = torch.stack([d['colvar'] for d in data])
    colvar = colvar[:, 0, 0]
    bins = 100
    x = torch.linspace(-pi, pi, bins)
    y = torch.histc(colvar, bins=bins, min=-pi, max=pi).log()
    plt.plot(x, y)
    plt.show()


def col(data):
    tim = torch.stack([d[p.tim] for d in data])
    colvar = torch.stack([d['colvar'] for d in data])
    plt.plot(tim[:, 0] / fs, colvar[:, 0, 0], '.')
    plt.plot(tim[:, 0] / fs, colvar[:, 0, 1], '.')
    plt.show()


def kbt(data):
    tim = torch.stack([d[p.tim] for d in data])
    kbt_ = torch.stack([d[p.kbt] for d in data])
    plt.plot(tim[:, 0] / fs, kbt_[:, 0] / kB)
    plt.show()


if __name__ == '__main__':
    main()
