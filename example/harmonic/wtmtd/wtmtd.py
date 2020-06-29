import math
from typing import Dict
from decimal import Decimal
from pathlib import Path
import torch
from torch import nn, Tensor
from ase.units import fs, kB
import torchfes as fes
import pnpot
import pointneighbor as pn


class ColVar(nn.Module):
    def __init__(self):
        super().__init__()
        self.pbc = torch.tensor([math.inf])

    def forward(self, inp: Dict[str, Tensor]):
        ret = inp[fes.p.pos][:, 0, 0][:, None]
        assert ret.size() == (1, 1)
        return ret


def make_inp():
    n_atm = 1
    n_bch = 1
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
    trj_path = Path('trj_wtmtd.pt')
    idx_path = Path('idx_wtmtd.pkl')
    hil_path = Path('hil_wtmtd.pt')
    hdx_path = Path('hdx_wtmtd.pkl')
    trn = fes.fes.mtd.meta_dynamics_to_well_tempared_metadynamics
    gam = torch.tensor(2.0)
    if trj_path.is_file():
        with fes.rec.open_torch(trj_path, 'rb', idx_path) as f:
            mol = f[-1]
        with fes.rec.open_torch(hil_path, 'rb') as f:
            for data in f:
                fes.fes.mtd.add_gaussian(mol, trn(data, gam))
        mode = 'ab'
    else:
        mol = make_inp()
        mode = 'wb'
    col = ColVar()
    res = fes.fes.mtd.GaussianPotential(col)
    mdl = pnpot.classical.Quadratic(torch.tensor([1.0]))
    eng = fes.ff.EvalEnergies(mdl, res)
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    mtd = fes.fes.mtd.WellTemparedMetaDynamics(col, [0.1], 0.01, gam, True)
    kbt = fes.md.GlobalLangevin()
    dyn = fes.md.PTPQ(eng, adj, kbt)
    with fes.rec.open_torch(trj_path, mode, idx_path) as rec,\
            fes.rec.open_torch(hil_path, mode, hdx_path) as hil:
        for i in range(100000):
            if i % 100 == 0:
                mol, new = mtd(mol)
                hil.write(new)
                stp = mol[fes.p.stp].item()
                hgt = round(Decimal(new[fes.p.mtd_hgt].item()), 4)
                print(f'{stp} {hgt}')
            mol = dyn(mol)
            rec.write(mol)


if __name__ == "__main__":
    main()
