from itertools import count
from typing import Dict
from decimal import Decimal
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torchani
from ignite.handlers.timing import Timer
from ase.io import read
from ase.units import fs, kB, Ha
from pnpot.nn.torchani import from_torchani
import pointneighbor as pn
import torchfes as fes
from torchfes.data.collate import ToDictTensor
from torchfes import forcefield as ff, md, inp, properties as p, api
from torchfes.recorder.torch import open_torch_mp
from torchfes.recorder import XYZRecorder
from torchfes.colvar.dihedral import Dihedral


def read_mol():
    atoms = read('diala.xyz')
    atoms.cell = np.eye(3) * 40.0
    loader = DataLoader([atoms],
                        batch_size=1,
                        collate_fn=ToDictTensor(['H', 'C', 'N', 'O']))
    return next(iter(loader))


def main():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    mol: Dict[str, Tensor] = read_mol()
    mol = {key: val.to(device) for key, val in mol.items()}
    inp.add_nvt(mol, 0.5 * fs, 300 * kB)
    inp.add_global_langevin(mol, 100.0 * fs)
    model_torchani = torchani.models.ANI1ccx()
    mdl = api.Unit(from_torchani(model_torchani, p.coo), Ha)
    num_dihed = torch.tensor([[5, 7, 9, 15]]).t() - 1
    colvar = Dihedral(num_dihed)
    mtd = fes.fes.mtd_new.MetaDynamics(colvar, [0.5], 0.02)
    res = fes.fes.mtd_new.GaussianPotential(colvar)
    eng = ff.EvalEnergies(mdl, res)
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

    dyn = md.PQP(eng, adj).to(device)
    timer = Timer()
    flush_interval = 200
    with open_torch_mp('trj.pt', 'wb') as rec, \
            open_torch_mp('hills.pt', 'wb') as hl, \
            XYZRecorder('xyz', 'w', model_torchani.species, 1) as xyz:
        for i in range(20000):
            if i % 100 == 0:
                mol, new = mtd(mol)
                hl.write(new)
            mol = dyn(mol)
            rec.write(mol)

            flush = i % flush_interval == flush_interval - 1
            eng = round(Decimal(mol[p.eng].item()), 5)
            eng_mol = round(Decimal(mol[p.eng_mol].item()), 5)
            eng_res = round(Decimal(mol[p.eng_res].item()), 5)
            tim = round(Decimal(timer.value()), 3)
            print(f'{i} {eng} {eng_mol} {eng_res} {tim}', flush=flush)
            timer.reset()
            xyz.append(mol)


if __name__ == "__main__":
    main()
