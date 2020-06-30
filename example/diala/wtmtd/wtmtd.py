import argparse
from decimal import Decimal
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchani
from ase.io import read
from ase.units import fs, kB, Ha
from pnpot.nn.torchani import from_torchani
import pointneighbor as pn
import torchfes as fes
from torchfes.data.collate import ToDictTensor
from torchfes import forcefield as ff, md, inp, properties as p, api
from torchfes.recorder import open_torch
from torchfes.colvar.dihedral import Dihedral


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


def setup_model(gam):
    idx = torch.tensor([[5, 7, 9, 15]]).t() - 1
    colvar = Dihedral(idx)
    mtd = fes.fes.mtd.WellTemparedMetaDynamics(colvar, [0.5], 0.02, gam, True)
    res = fes.fes.mtd.GaussianPotential(colvar)
    model_torchani = torchani.models.ANI1ccx()
    mdl = api.Unit(from_torchani(model_torchani, p.coo), Ha)
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
    return eng, adj, mtd


def main():
    gam = 40.0
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', action='store_true')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    trj_path = fes.rec.PathPair('trj')
    hil_path = fes.rec.PathPair('hil')
    mol, cont = fes.inp.continue_inp(init_mol, trj_path, device, args.init)
    mode = 'a' if cont else 'w'
    if cont:
        mol = fes.rec.read_mtd(hil_path, mol)
        mol = fes.fes.mtd.mtd_to_wtmtd(mol, gam)
    eng, adj, mtd = setup_model(gam)

    dyn = md.PTPQ(eng, adj, fes.md.GlobalLangevin()).to(device)
    with open_torch(trj_path, mode) as rec, open_torch(hil_path, mode) as hl:
        for i in range(100000):
            if i % 100 == 0:
                mol, new = mtd(mol)
                hl.write(new)
                hgt = ' '.join([
                    f'{round(Decimal(v), 10)}'
                    for v in new[p.mtd_hgt].tolist()
                ])
                stp = int(mol[p.stp].item())
                print(stp, hgt, flush=True)

            mol = dyn(mol)
            rec.write(fes.rec.not_tmp(mol))


if __name__ == "__main__":
    main()
