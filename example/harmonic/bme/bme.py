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
    def __init__(self):
        super().__init__()
        self.pbc = torch.tensor([math.inf])

    def forward(self, inp: Dict[str, Tensor]):
        ret = inp[fes.p.pos][:, 0, 0][:, None]
        assert ret.size(1) == 1
        return fes.colvar.add_colvar(inp, ret)


def make_inp():
    n_atm = 1
    n_bch = 30
    n_dim = 1
    cel = torch.eye(n_dim)[None, :, :].expand((n_bch, n_dim, n_dim)) * 1000
    pos = torch.rand([n_bch, n_atm, n_dim])
    pbc = torch.zeros([n_bch, n_dim], dtype=torch.bool)
    elm = torch.ones([n_bch, n_atm])
    mas = torch.ones([n_bch, n_atm])
    inp = fes.inp.init_mol(cel, pbc, elm, pos, mas)
    fes.inp.add_nvt(inp, 1.0 * fs, 300 * kB)
    fes.inp.add_global_langevin(inp, 100.0 * fs)
    return inp


def main():
    pre_path = Path('pre')
    trj_path = Path('trj')
    with fes.rec.TorchTrajectory(pre_path, 'rb') as f:
        mol = f[-1]
        mol.pop(fes.p.rst)
    mode = 'wb'
    mdl = pnpot.classical.Quadratic(torch.tensor([1.0]))
    my_colvar = MyColVar()
    col = fes.colvar.ColVar([my_colvar])
    n_bch = mol[fes.p.pos].size(0)
    mol[fes.p.col_cen] = torch.linspace(-0.5, 0.5, n_bch)[:, None]
    eng = fes.ff.EvalEnergies(mdl, col=col)
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    kbt = fes.md.GlobalLangevin()
    dyn = fes.md.PTPQs(eng, adj, kbt, col, 1e-7, True)
    timer = ignite.handlers.Timer()
    with fes.rec.open_trj(trj_path, mode) as rec:
        for _ in range(1000):
            mol = dyn(mol)
            rec.put(fes.utils.detach(fes.data.filter_case(mol, fes.p.saves)))
            tim = round(Decimal(timer.value()), 3)
            timer.reset()
            print(f'{_} {tim}')
            if mol[fes.p.frc].abs().max() < 1e-4:
                break


if __name__ == "__main__":
    main()
