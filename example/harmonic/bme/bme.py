from pathlib import Path
import math
from typing import Dict
from decimal import Decimal
import torch
from torch import nn, Tensor
from ase.units import fs, kB
import ignite
import torchfes as fes
import pnpot
import pointneighbor as pn


class ColVar(nn.Module):
    def __init__(self):
        super().__init__()
        self.pbc = torch.tensor([math.inf])

    def forward(self, inp: Dict[str, Tensor]):
        ret = inp[fes.p.pos][:, 0, 0][:, None]
        return ret


def make_inp():
    n_atm = 1
    n_bch = 30
    n_dim = 1
    cel = torch.eye(n_dim)[None, :, :].expand((n_bch, n_dim, n_dim)) * 1000
    pos = torch.rand([n_bch, n_atm, n_dim])
    pbc = torch.zeros([n_bch, n_dim], dtype=torch.bool)
    elm = torch.ones([n_bch, n_atm])
    mas = torch.ones([n_bch, n_atm])
    inp = fes.mol.init_mol(cel, pbc, elm, pos, mas)
    inp = fes.mol.add_nvt(inp, 1.0 * fs, 300 * kB)
    inp = fes.mol.add_global_langevin(inp, 100.0 * fs)
    return inp


def main():
    pre_path = Path('pre')
    trj_path = Path('trj')
    with fes.rec.TorchTrajectory(pre_path, 'rb') as f:
        mol = f[-1]
    mode = 'wb'
    mdl = pnpot.classical.Quadratic(torch.tensor([1.0]))
    col = ColVar()
    n_bch = mol[fes.p.pos].size(0)
    mol[fes.p.con_cen] = torch.linspace(-0.5, 0.5, n_bch)[:, None]
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    evl = fes.ff.get_adj_eng_frc(adj, mdl)
    kbt = fes.md.GlobalLangevin()
    dyn = fes.md.leap_frog(evl, kbt, col)
    timer = ignite.handlers.Timer()
    mol = evl(mol)
    with fes.rec.open_trj(trj_path, mode) as rec:
        for _ in range(1000):
            mol = dyn(mol)
            rec.put(fes.data.filter_case(mol, fes.p.saves))
            tim = round(Decimal(timer.value()), 3)
            timer.reset()
            print(f'{_} {tim}')
            if mol[fes.p.frc].abs().max() < 1e-4:
                break


if __name__ == "__main__":
    main()
