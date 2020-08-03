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


class MyColVar(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.pbc = torch.tensor([math.inf])
        self.i = i

    def forward(self, inp: Dict[str, Tensor]):
        ret = inp[fes.p.pos][:, 0, self.i][:, None]
        assert ret.size(1) == 1
        return fes.colvar.add_colvar(inp, ret)


def make_inp():
    n_atm = 1
    n_bch = 30
    n_dim = 2
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
    mol = make_inp()
    mode = 'wb'
    mdl = pnpot.classical.Quadratic(torch.tensor([1.0]))
    my_colvar1 = MyColVar(0)
    my_colvar2 = MyColVar(1)
    col = fes.colvar.ColVar([my_colvar1, my_colvar2])
    msk, _ = col[[my_colvar1, my_colvar2]]
    res = fes.res.Restraints([
        fes.res.HarmonicRestraints(msk, torch.tensor([0]), torch.tensor([5.0]))
    ])
    n_bch = mol[fes.p.pos].size(0)
    mol[fes.p.col_cen] = torch.linspace(-0.5, 0.5, n_bch)[:, None]
    eng = fes.ff.EvalEnergies(mdl, col, res)
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    dyn = fes.md.PQF(eng, adj, fes.md.FIRE(0.5 * fs))
    timer = ignite.handlers.Timer()
    with fes.rec.open_trj(pre_path, mode) as rec:
        for _ in range(100):
            mol = dyn(mol)
            rec.put(mol)
            tim = round(Decimal(timer.value()), 3)
            timer.reset()
            print(f'{mol[fes.p.col_var]} {tim}')
            if mol[fes.p.frc].abs().max() < 1e-4:
                break


if __name__ == "__main__":
    main()
