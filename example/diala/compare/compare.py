from typing import Dict
import numpy as np
import torch
from torch import Tensor, jit
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
from torchfes.recorder.torch import open_torch, PathPair


def read_mol():
    atoms = read('../diala.xyz')
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
    eng = ff.EvalEnergies(mdl)
    rc_r = mdl.mdl.aev.rad.rc
    rc_a = mdl.mdl.aev.ang.rc
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(rc_r),
        [
            (p.coo, rc_r),
            (p.coo, rc_a),
        ]
    )

    dyn = jit.script(md.PQP(eng, adj).to(device))
    eng = jit.script(eng.to(device))
    adj = jit.script(adj)
    timer = Timer()
    with open_torch(PathPair('compare'), 'wb') as rec:
        for i in range(2000000):
            mol = dyn(mol)
            timer.reset()
            mol = adj(mol)
            eng_pnpot = eng(mol)[p.eng_mol].item()
            time_pnpot = timer.value()
            rec.write(mol)
            timer.reset()
            eng_torch = model_torchani(
                (mol[p.elm], mol[p.pos]),
                mol[p.cel][0], mol[p.pbc][0]).energies.item()
            time_torch = timer.value()
            eng_pnpot = mol[p.eng].item()
            print(i, eng_pnpot, eng_pnpot - eng_torch, time_pnpot, time_torch)


if __name__ == "__main__":
    main()
