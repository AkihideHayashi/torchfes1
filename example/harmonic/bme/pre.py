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
    inp = fes.inp.init_inp(cel, pbc, elm, pos, mas)
    fes.inp.add_nvt(inp, 1.0 * fs, 300 * kB)
    fes.inp.add_global_langevin(inp, 100.0 * fs)
    return inp


def main():
    pre_path = fes.rec.PathPair('pre')
    if pre_path.is_file():
        with fes.rec.open_torch(pre_path, 'rb') as f:
            mol = f[-1]
        mode = 'ab'
    else:
        mol = make_inp()
        mode = 'wb'
    mdl = pnpot.classical.Quadratic(torch.tensor([1.0]))
    col = ColVar()
    res = fes.res.HarmonicRestraints(col,
                                     torch.tensor([0]), torch.tensor([5.0]))
    n_bch = mol[fes.p.pos].size(0)
    mol[fes.p.res_cen] = torch.linspace(-0.5, 0.5, n_bch)[:, None]
    eng = fes.ff.EvalEnergies(mdl, res)
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    dyn = fes.md.PQF(eng, adj, fes.md.FIRE(0.5, 5, 0.5, 1.1, 0.5, 0.5 * fs))
    timer = ignite.handlers.Timer()
    with fes.rec.open_torch(pre_path, mode) as rec:
        for _ in range(100):
            mol = dyn(mol)
            rec.write(mol)
            tim = round(Decimal(timer.value()), 3)
            timer.reset()
            col_var = col(mol)
            print(f'{col_var} {tim}')
            if mol[fes.p.frc].abs().max() < 1e-4:
                break


if __name__ == "__main__":
    main()
