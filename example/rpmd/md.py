import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from decimal import Decimal
import torch
from ase.units import fs, kB
import ignite
import torchfes as fes
import pnpot
import pointneighbor as pn

torch.set_default_dtype(torch.float64)

n_ring = 16
hbar = 1.0
k = 1.0
kbt = 0.1
beta = 10.0


def make_inp():
    n_atm = 1
    n_bch = n_ring
    n_dim = 1
    cel = torch.eye(n_dim)[None, :, :].expand((n_bch, n_dim, n_dim)) * 1000
    pos = torch.rand([n_bch, n_atm, n_dim])
    pbc = torch.zeros([n_bch, n_dim], dtype=torch.bool)
    elm = torch.ones([n_bch, n_atm])
    mas = torch.ones([n_bch, n_atm])
    inp = fes.mol.init_mol(cel, pbc, elm, pos, mas)
    inp = fes.mol.add_nvt(inp, 0.01, 0.1)
    inp = fes.mol.add_global_langevin(inp, 0.7)
    return inp


def make_inp_or_continue(path):
    if path.is_file():
        with fes.rec.open_torch(path, 'rb') as f:
            mol = f[-1]
        mode = 'ab'
    else:
        mol = make_inp()
        mode = 'wb'
    return mol, mode


def main():
    trj_path = Path('md')
    mol, mode = make_inp_or_continue(trj_path)
    mol[fes.p.dtm] = mol[fes.p.dtm].mean(dim=0, keepdim=True)
    mdl = pnpot.classical.Quadratic(torch.tensor([k]))
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    evl = fes.ff.get_adj_eng_frc(adj, mdl)
    mom = fes.md.UpdtMom(0.5)
    base = fes.md.rpmd_base(hbar, beta, n_ring, 'cpu')
    pos = fes.md.RpmdPos(base, mol[fes.p.mas], mol[fes.p.dtm].item(), 1)
    lan = fes.md.RpmdLangevin(
        base, 0.7, mol[fes.p.dtm].item() * 0.5, mol[fes.p.mas], 1)
    kin = fes.md.RpmdKin(base)
    # mol[fes.p.mom][:, 0, 0] = torch.tensor([-0.39253548, -0.23131893, -0.39253548, -0.23131893])
    mol = evl(mol)
    pos_lst = []
    pot_lst = []
    kin_lst = []
    mol[fes.p.mom][:, 0, 0] = torch.ones(n_ring)
    mol[fes.p.pos][:, 0, 0] = torch.zeros(n_ring)
    for i in range(40000):
        mol = lan(mol)
        # print(mol[fes.p.mom].flatten())
        mol = mom(mol)
        mol = pos(mol)
        mol = evl(mol)
        mol = kin(mol)
        mol = mom(mol)
        mol = lan(mol)
        # print(mol[fes.p.mom].flatten())
        # print(mol[fes.p.pos].flatten())
        pos_lst.append(mol[fes.p.pos][:, 0, 0])
        pot_lst.append(mol[fes.p.eng][:])
        kin_lst.append(mol[fes.p.rpm_kin])
        print(i)
    with open('tmp.pkl', 'wb') as f:
        pickle.dump([pos_lst, pot_lst, kin_lst], f)


def gound(x, hbar, m, k):
    omega = torch.sqrt(k / m)
    xi = torch.sqrt(m * omega / hbar) * x
    return torch.exp(-0.5 * xi * xi)


if __name__ == "__main__":
    main()
