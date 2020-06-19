from math import pi
from typing import Dict
import torch
from torch import nn, Tensor
from ase.units import kB, fs, Ha, Bohr, Ang, AUT
import pointneighbor as pn
from pointneighbor import AdjSftSpc
from pnpot.classical import LennardJones
from torchfes.forcefield import EvalEnergies, EvalEnergiesForces
from torchfes.utils import sym_to_elm
from torchfes.inp import init_inp, add_nvt, add_global_nose_hoover_chain
from torchfes import md
from torchfes import properties as p
from torchfes.recorder.xyz import XYZRecorder
import torchfes.functional as fn


class Fix12(nn.Module):
    def forward(self, inp: Dict[str, Tensor], _: AdjSftSpc):
        pos = inp[p.pos]
        rr1 = pos[:, 0, :]
        rr2 = pos[:, 1, :]
        rr12 = rr2 - rr1
        r12 = rr12.norm(2, -1)[:, None]
        return r12 - 3.405


class Fix213(nn.Module):
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
        return cos.acos()[:, None] - pi / 2


def parse_vel(xyz):
    return torch.tensor([[
        [float(w) for w in line.split()[1:]] for line in xyz.split('\n')
        if line.strip()
    ]])

# print(parse_vel("""
#  Ar         0.0021033285       -0.0005843461        0.0006596157
#  Ar        -0.0000369425        0.0005716938        0.0002171618
#  Ar        -0.0020663860        0.0000126523       -0.0008767775
#  """) * Ang / fs)


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
    # add_md(inp, 1.0 * fs, 85.0 * kB)
    add_nvt(inp, 1.0 * fs, 85.0 * kB)
    add_global_nose_hoover_chain(inp, (1000.0 * fs))

    vel = torch.tensor([[[0.0214, -0.0059, 0.0067],
                         [-0.0004, 0.0058, 0.0022],
                         [-0.0210, 0.0001, -0.0089]]])
    for v in vel[0] / Bohr * AUT:
        print('{} {} {}'.format(*v))
    mom = vel * mas[:, :, None]
    inp[p.mom] = mom
    return inp


def calc(eng, adj, inp):
    return EvalEnergiesForces(eng)(
        inp, adj(pn.pnt_ful(inp[p.cel], inp[p.pbc], inp[p.pos], inp[p.ent])))


def main():
    torch.set_default_dtype(torch.float64)
    inp = make_inp()
    eng = EvalEnergies(
        LennardJones(e=torch.tensor([0.1]), s=torch.tensor([3.405]))
    )
    adj = pn.Coo2FulSimple(1000.0)
    con = Fix213()
    kbt = md.unified.GlobalNHC(1)
    # dyn = md.PQ(eng, adj)
    # dyn = md.PQS(eng, adj, con, tol=1e-5)
    # dyn = md.PTPQ(eng, adj, kbt)
    dyn = md.PTPQS(eng, adj, kbt, con, 1e-8)
    # dyn = md.TPQSPTR(eng, adj, kbt, con, 1e-5, 1e-5)

    kin_ = fn.kinetic_energies(inp[p.mom], inp[p.mas], inp[p.ent])
    tem = fn.temperatures(kin_, inp[p.ent], 3)
    xyz = XYZRecorder('xyz', 'w', ['Ar'], 1)
    inp = calc(eng, adj, inp)
    print('initial kin [Ha] Temp[K]')
    print(f'{kin_[0].item() / Ha : < 10.8} {tem[0].item() / kB :< 10.8}')
    print(inp[p.eng_mol][0].item() / Ha)
    print(
        'Step Nr.    Time[fs]   Kin.[a.u.]       Temp[K]         Pot.[a.u.]'
        '        Cons Qty[a.u.]        UsedTime[s]')
    kin = fn.kinetic_energies(inp[p.mom], inp[p.mas], inp[p.ent])
    tem = fn.temperatures(kin, inp[p.ent], 3)
    tim = inp[p.tim][0].item() / fs
    kin = kin[0].item() / Ha
    tem = tem[0].item() / kB
    pot = inp[p.eng_mol][0].item() / Ha
    print(f'{0:2}        {tim:8.6}    {kin:<10.4}   {tem}   {pot}')
    for i in range(100):
        inp = dyn(inp)
        kin = fn.kinetic_energies(inp[p.mom], inp[p.mas], inp[p.ent])
        tem = fn.temperatures(kin, inp[p.ent], 3)
        con = kin + inp[p.eng_mol]
        tim = inp[p.tim][0].item() / fs
        kin = kin[0].item() / Ha
        tem = tem[0].item() / kB
        pot = inp[p.eng_mol][0].item() / Ha
        con = con[0].item() / Ha
        # print(f'{i:2}        {tim:8.6}    {kin:<10.8}   {tem}   {pot}   {con}')
        # print(inp[p.bme_lmd] / Ha * Bohr)
        # print(((inp[p.bme_lmd] + inp[p.kbt] * inp[p.bme_cor]) / Ha).item())
        print((inp[p.bme_lmd] / Ha).item())
        xyz.append(inp)


if __name__ == "__main__":
    main()
