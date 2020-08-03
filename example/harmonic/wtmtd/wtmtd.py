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
    n_bch = 1
    n_dim = 1
    cel = torch.eye(n_dim)[None, :, :].expand((n_bch, n_dim, n_dim)) * 1000
    pos = torch.rand([n_bch, n_atm, n_dim])
    pbc = torch.zeros([n_bch, n_dim], dtype=torch.bool)
    elm = torch.ones([n_bch, n_atm])
    mas = torch.ones([n_bch, n_atm])
    inp = fes.mol.init_mol(cel, pbc, elm, pos, mas)
    inp = fes.mol.add_nvt(inp, 1.0 * fs, 300 * kB)
    inp = fes.mol.add_global_langevin(inp, 100.0 * fs)
    inp = fes.data.reprecate(inp, 3)
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
    my_colvar = MyColVar()
    col = fes.colvar.ColVar([my_colvar])
    res = fes.fes.mtd.GaussianPotential(*col[[my_colvar]])
    mdl = pnpot.classical.Quadratic(torch.tensor([1.0]))
    eng = fes.ff.EvalEnergies(mdl, col, res)
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    bias_factor = 2.0
    mtd = fes.fes.mtd.EnsembleMTD(
        fes.fes.mtd.WellTemparedMetaDynamics(
            *col[[my_colvar]], [0.1], 0.01, bias_factor))
    kbt = fes.md.GlobalLangevin()
    dyn = fes.md.PTPQ(eng, adj, kbt)
    timer = ignite.handlers.Timer()
    with fes.rec.open_trj(trj_path, mode) as rec,\
            fes.rec.open_trj(hil_path, mode) as hil:
        for i in range(10000):
            mol = dyn(mol)
            if i % 100 == 0:
                mol, new = mtd(mol)
                hil.put(fes.fes.wtmtd_to_mtd(new, bias_factor))
            rec.put(mol)
            stp = mol[fes.p.stp].tolist()
            tim = round(Decimal(timer.value()), 3)
            eng = '[{}]'.format(
                ' '.join([
                    tostr(x) for x in mol[fes.p.eng_res][:, 0].tolist()]))
            timer.reset()
            print(f'{stp} {eng} {tim}')


def tostr(x):
    return f'{round(Decimal(x), 5)}'


if __name__ == "__main__":
    main()