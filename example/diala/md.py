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
from torchfes.recorder.torch import TorchRecorder


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
    mdl = api.Unit(from_torchani(model_torchani), Ha)
    eng = ff.EvalEnergies(mdl)
    rc = mdl.mdl.aev.rad.rc
    delta = 1.0
    adj = pn.Coo2BookKeeping(
        pn.Coo2FulSimple(rc + delta), pn.StrictCriteria(delta), rc)
    dyn = jit.script(md.PQP(eng, adj).to(device))
    # num_dihed = torch.tensor([[11, 9, 15, 17], [11, 9, 7, 5]]) - 1
    # colvar = Dihedral(num_dihed)
    timer = Timer()
    flush_interval = 200
    with TorchRecorder('trj.pt', 'wb') as rec:
        for i in range(2000000):
            mol = dyn(mol)
            rec.append(mol)
            print(i, mol[p.eng].item(), timer.value(),
                  flush=i % flush_interval == flush_interval-1)
            timer.reset()


if __name__ == "__main__":
    main()
