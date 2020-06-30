from decimal import Decimal
import numpy as np
import torch
from torch import jit
from torch.utils.data import DataLoader
import torchani
from ignite.handlers.timing import Timer
from ase.io import read
from ase.units import fs, kB, Ha
from pnpot.nn.torchani import from_torchani
import pointneighbor as pn
from torchfes.data.collate import ToDictTensor
from torchfes import forcefield as ff, md, inp, properties as p, api
import torchfes as fes
from torchfes.recorder import open_torch, PathPair


def init_mol():
    atoms = read('../diala.xyz')
    atoms.cell = np.eye(3) * 40.0
    loader = DataLoader([atoms],
                        batch_size=1,
                        collate_fn=ToDictTensor(['H', 'C', 'N', 'O']))
    mol = next(iter(loader))
    inp.add_nvt(mol, 0.5 * fs, 300 * kB)
    inp.add_global_langevin(mol, 100.0 * fs)
    return mol


def setup_model():
    model_torchani = torchani.models.ANI1ccx()
    mdl = api.Unit(from_torchani(model_torchani, p.coo), Ha)
    eng = ff.EvalEnergies(mdl)
    rc_r = mdl.mdl.aev.rad.rc
    rc_a = mdl.mdl.aev.ang.rc
    delta = 1.0
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2BookKeeping(
            pn.Coo2FulSimple(rc_r + delta), pn.StrictCriteria(delta), rc_r),
        [
            (p.coo, rc_r),
            (p.coo, rc_a),
        ]
    )
    return eng, adj


def main():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    trj_dir = PathPair('md')
    mol, mode = fes.inp.continue_inp(init_mol, trj_dir, device, False)
    eng, adj = setup_model()
    dyn = jit.script(md.TPQPT(eng, adj, md.GlobalLangevin()).to(device))
    timer = Timer()
    flush_interval = 200
    with open_torch(trj_dir, mode) as rec:
        for i in range(20000):
            mol = dyn(mol)
            rec.write(mol)
            eng = round(Decimal(mol[p.eng].item()), 7)
            flush = i % flush_interval == flush_interval - 1
            real_time = round(Decimal(timer.value()), 4)
            stp = int(mol[fes.p.stp].item())
            print(f'{stp} {eng:>10} {real_time:>5}', flush=flush)
            timer.reset()


if __name__ == "__main__":
    main()
