import os
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
    pos = torch.rand([n_bch, n_atm, n_dim]) * 0.01
    pbc = torch.zeros([n_bch, n_dim], dtype=torch.bool)
    elm = torch.ones([n_bch, n_atm])
    mas = torch.ones([n_bch, n_atm])
    inp = fes.mol.init_mol(cel, pbc, elm, pos, mas)
    inp = fes.mol.add_basic(inp)
    inp = fes.mol.add_nvt(inp, 1.0 * fs, 300 * kB)
    inp = fes.mol.add_global_langevin(inp, 100.0 * fs)
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


class Callback:
    def __init__(self, trj_path) -> None:
        self.timer = ignite.handlers.Timer()
        self.rec = fes.rec.open_trj(trj_path, 'ab')

    def __call__(self, i, mol):
        if mol is StopIteration:
            self.rec.put(mol)
        else:
            self.rec.put(fes.data.filter_case(mol, fes.p.saves))
            print(i, mol[fes.p.stp], self.timer.value())
            self.timer.reset()


def main():
    os.environ["PLUMED_KERNEL"] = "/Users/akihide/.local/lib/libplumedKernel.dylib"
    trj_path = Path('trj')
    mol, mode = make_inp_or_continue(trj_path)
    col = ColVar()
    res = fes.fes.mtd.GaussianPotential(col)
    mdl = pnpot.classical.Quadratic(torch.tensor([1.0]))
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    evl = fes.ff.get_adj_eng_frc(adj, mdl, ext_eng=res)
    plumed = fes.fes.Plumed(1, mol[fes.p.kbt].item(), mol[fes.p.dtm].item())
    plumed.readInputLine("RESTART")
    plumed.readInputLine("p: POSITION ATOM=1")
    plumed.readInputLine("metad: METAD ARG=p.x,p.y PACE=100 HEIGHT=1.0 SIGMA=0.1,0.1 GRID_MIN=-5.0,-5.0 GRID_MAX=5.0,5.0 BIASFACTOR=100.0 CALC_RCT GRID_WFILE=HILL GRID_RFILE=HILL GRID_WSTRIDE=1000")
    plumed.readInputLine("PRINT ARG=p.x,metad.* FILE=COLVAR")
    kbt = fes.md.GlobalLangevin()
    callback = Callback(trj_path)
    fes.md.dynamics.velocity_verlet_plumed(
        mol, evl, kbt, plumed, callback, 10000)
    callback(-1, StopIteration)


if __name__ == "__main__":
    main()
