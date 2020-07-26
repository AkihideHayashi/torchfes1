from pathlib import Path
from decimal import Decimal
import torch
from ase.units import fs, kB
import ignite
import torchfes as fes
import pnpot
import pointneighbor as pn


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
    mdl = pnpot.classical.Quadratic(torch.tensor([1.0]))
    eng = fes.ff.EvalEnergies(mdl)
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    kbt = fes.md.GlobalLangevin()
    dyn = fes.md.PTPQ(eng, adj, kbt)
    timer = ignite.handlers.Timer()
    with fes.rec.open_trj(trj_path, mode) as rec:
        for _ in range(10000):
            mol = dyn(mol)
            rec.put(mol)
            stp = mol[fes.p.stp].item()
            tim = round(Decimal(timer.value()), 3)
            timer.reset()
            print(f'{stp} {tim}')


if __name__ == "__main__":
    main()
