import math
from pathlib import Path
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
        assert ret.size(1) == 1
        return ret


def make_inp():
    n_atm = 1
    n_bch = 1
    n_dim = 3
    cel = torch.eye(n_dim)[None, :, :].expand((n_bch, n_dim, n_dim)) * 1000
    pos = torch.rand([n_bch, n_atm, n_dim])
    pbc = torch.zeros([n_bch, n_dim], dtype=torch.bool)
    elm = torch.ones([n_bch, n_atm])
    mas = torch.ones([n_bch, n_atm])
    inp = fes.mol.init_mol(cel, pbc, elm, pos, mas)
    inp = fes.mol.add_basic(inp)
    inp = fes.mol.add_nvt(inp, 1.0 * fs, 300 * kB)
    inp = fes.mol.add_global_langevin(inp, 100.0 * fs)
    inp = fes.data.reprecate(inp, 3)
    inp[fes.p.idt] = torch.arange(n_bch)
    return inp


def make_inp_or_continue(path: Path):
    if path.is_file():
        with fes.rec.open_trj(path, 'rb') as f:
            mol = f[-1]
        mode = 'ab'
    else:
        mol = make_inp()
        mode = 'wb'
    return mol, mode


def main():
    trj_path = Path('trj')
    hil_path = Path('hil')
    mol, mode = make_inp_or_continue(trj_path)
    col = ColVar()
    res = fes.fes.mtd.GaussianPotential(col)
    mdl = pnpot.classical.Quadratic(torch.tensor([1.0]))
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    evl = fes.ff.get_adj_eng_frc(adj, mdl, ext_eng=res)
    mtd = fes.fes.mtd.EnsembleMTD(
        fes.fes.mtd.WellTemparedMetaDynamics(col, [0.1], 0.01, 2.0))
    kbt = fes.md.GlobalLangevin()
    dyn = fes.md.leap_frog(evl, kbt=kbt)
    timer = ignite.handlers.Timer()
    mol = evl(mol)
    with fes.rec.open_trj(trj_path, mode) as rec,\
            fes.rec.open_trj(hil_path, mode) as hil:
        for i in range(10000):
            if i % 100 == 0:
                mol, new = mtd(mol)
                hil.put(fes.fes.mtd.wtmtd_to_mtd(new, mtd.new.gam))
                stp = mol[fes.p.stp].tolist()
                print(stp, new[fes.p.mtd_hgt].tolist())
                # hil.put(fes.fes.mtd.wtmtd_to_mtd(new, mtd.new.gam))
            mol = dyn(mol)
            rec.put(mol)
            stp = mol[fes.p.stp].tolist()
            timer.reset()
            # print(f'{stp} {eng} {tim}')


def tostr(x):
    return f'{round(Decimal(x), 5)}'


if __name__ == "__main__":
    main()