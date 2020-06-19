from math import pi
from typing import Dict
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torchani
from ase.io import read
from ase.units import fs, kB, Ha
from pnpot.nn.torchani import from_torchani
import pointneighbor as pn
from torchfes.data.collate import ToDictTensor
from torchfes import forcefield as ff, md, inp, properties as p, api
from torchfes.recorder import XYZRecorder, hdf5_recorder


def dihedral(pos, num):
    r = pos[:, num, :]
    rr10 = r[:, :, 1, :] - r[:, :, 0, :]
    rr23 = r[:, :, 2, :] - r[:, :, 3, :]
    r10 = rr10.norm(2, -1)
    r23 = rr23.norm(2, -1)
    return ((rr10 * rr23).sum(-1) / (r10 * r23)).acos()


class Dihedral(nn.Module):
    def __init__(self, num):
        super().__init__()
        self.num = num

    def forward(self, inp: Dict[str, Tensor], _: pn.AdjSftSpc):
        return dihedral(inp[p.pos], self.num)[:, None]


def read_mol():
    atoms = read('diala.xyz')
    atoms.cell = np.eye(3) * 40.0
    loader = DataLoader([atoms],
                        batch_size=1,
                        collate_fn=ToDictTensor(['H', 'C', 'N', 'O']))
    return next(iter(loader))


def main():
    mol = read_mol()
    inp.add_nvt(mol, 0.5 * fs, 300 * kB)
    inp.add_global_langevin(mol, 100.0 * fs)
    model_torchani = torchani.models.ANI1ccx()
    order = model_torchani.species
    mdl = api.Unit(from_torchani(model_torchani), Ha)
    eng = ff.EvalEnergies(mdl)
    adj = pn.Coo2FulSimple(mdl.mdl.aev.rad.rc)
    dyn = md.PQP(eng, adj)
    # num_dihed = torch.tensor([[11, 9, 15, 17], [11, 9, 7, 5]]) - 1
    # colvar = Dihedral(num_dihed)
    with XYZRecorder('xyz', 'w', order, 1) as xyz, \
            hdf5_recorder('trj.hdf5', 'w') as hdf:
        for i in range(4000):
            mol = dyn(mol)
            print(i, mol[p.eng].item())
            xyz.append(mol)
            hdf.append(mol)


if __name__ == "__main__":
    main()
