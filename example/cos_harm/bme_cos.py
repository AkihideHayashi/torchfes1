from typing import Dict
import torch
from torch import nn, Tensor
from ase.units import kB, fs, Ang
import pointneighbor as pn
from pointneighbor import AdjSftSpc
from torchfes.forcefield import EvalEnergies, EvalEnergiesForces
from torchfes.utils import sym_to_elm
from torchfes.inp import init_inp, add_nvt, add_global_langevin
from torchfes import md, api, data
from torchfes import properties as p
from torchfes.restraints import HarmonicRestraints
from torchfes.colvar import ColVarSft
from torchfes.recorder import hdf5_recorder


class Potential(nn.Module):
    def __init__(self, k, r0):
        super().__init__()
        self.k = k
        self.r0 = r0

    def forward(self, inp: Dict[str, Tensor], _: AdjSftSpc):
        pos = inp[p.pos]
        rr1 = pos[:, 0, :]
        rr2 = pos[:, 1, :]
        rr3 = pos[:, 2, :]
        rr12 = rr2 - rr1
        rr13 = rr3 - rr1
        r12 = rr12.norm(2, -1)
        r13 = rr13.norm(2, -1)
        eng_mol = self.k * ((r12 - self.r0) ** 2 + (r13 - self.r0) ** 2)
        eng_atm = torch.zeros_like(inp[p.elm], dtype=pos.dtype)

        return api.Energies(eng_mol=eng_mol, eng_atm=eng_atm,
                            eng_mol_std=torch.zeros_like(eng_mol),
                            eng_atm_std=torch.zeros_like(eng_atm))


class Cos213(nn.Module):
    def forward(self, inp: Dict[str, Tensor], _: AdjSftSpc):
        pos = inp[p.pos]
        rr1 = pos[:, 0, :]
        rr2 = pos[:, 1, :]
        rr3 = pos[:, 2, :]
        rr12 = rr2 - rr1
        rr13 = rr3 - rr1
        r12 = rr12.norm(2, -1)
        r13 = rr13.norm(2, -1)
        cos = (rr12 * rr13).sum(-1) / (r12 * r13)
        return cos[:, None]


def make_inp():
    order = ['Ar']
    sym = [['Ar', 'Ar', 'Ar']]
    cel = torch.eye(3)[None, :] * 200.0 * Ang
    pbc = torch.tensor([[True, True, True]])
    pos = torch.tensor([[
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 8.405],
        [5.0, 8.405, 5.0],
    ]]) * Ang
    elm = torch.tensor(sym_to_elm(sym, order))
    mas = torch.tensor([40.0])[elm]
    inp = init_inp(cel, pbc, elm, pos, mas)
    add_nvt(inp, 1.0 * fs, 85.0 * kB)
    add_global_langevin(inp, 10.0 * fs)
    return data.reprecate(inp, 10)


def calc(eng, adj, inp):
    return EvalEnergiesForces(eng)(
        inp, adj(pn.pnt_ful(inp[p.cel], inp[p.pbc], inp[p.pos], inp[p.ent])))


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(linewidth=200)
    inp = make_inp()
    pot = Potential(1.0, 3.405)
    colvar = Cos213()
    collective = ColVarSft(colvar, torch.linspace(-0.9, 0.9, inp[p.elm].size(0))[:, None])
    eng = EvalEnergies(
        pot,
        HarmonicRestraints(
            collective, torch.tensor([0]), torch.tensor([100.0]))
    )
    adj = pn.Coo2FulSimple(100.0)
    kbt = md.unified.GlobalLangevin()
    dyn = md.PTPQ(eng, adj, kbt)

    print('start pre md')
    for i in range(1000):
        inp = dyn(inp)
    print('end pre md')

    eng = EvalEnergies(pot)
    dyn = md.PTPQS(eng, adj, kbt, collective, 5e-4)
    # dyn = md.TPQSPTR(eng, adj, kbt, collective, 5e-4, 5e-4)
    with hdf5_recorder('bme.hdf5', 'w') as rec:
        for i in range(50000):
            inp = dyn(inp)
            print(i, '/ 50000')
            rec.append(inp)




if __name__ == "__main__":
    main()
