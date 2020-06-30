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
    trj_path = fes.rec.PathPair('trj')
    with fes.rec.open_torch(pre_path, 'rb') as f:
        mol = f[-1]
        mol.pop(fes.p.sld_rst)
    mode = 'wb'
    mdl = pnpot.classical.Quadratic(torch.tensor([1.0]))
    col = ColVar()
    n_bch = mol[fes.p.pos].size(0)
    mol[fes.p.bme_cen] = torch.linspace(-0.5, 0.5, n_bch)[:, None]
    eng = fes.ff.EvalEnergies(mdl)
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    kbt = fes.md.GlobalLangevin()
    dyn = fes.md.PTPQs(eng, adj, kbt, col, 1e-7, True)
    timer = ignite.handlers.Timer()
    with fes.rec.open_torch(trj_path, mode) as rec:
        for _ in range(1000):
            mol = dyn(mol)
            rec.write({key: val.clone().detach() for key, val
                       in fes.rec.selector.not_tmp(mol).items()})
            tim = round(Decimal(timer.value()), 3)
            timer.reset()
            print(f'{_} {tim}')
            if mol[fes.p.frc].abs().max() < 1e-4:
                break


if __name__ == "__main__":
    main()
