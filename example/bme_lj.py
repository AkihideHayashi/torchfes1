import argparse
import decimal
from math import pi
from typing import Dict
import torch
from torch import nn, Tensor, jit
from ase.units import kB, fs, Ha, Bohr, Ang, AUT
import ignite
import pointneighbor as pn
from pnpot.classical import LennardJones
from torchfes.forcefield import EvalEnergies
from torchfes.utils import sym_to_elm
from torchfes.inp import init_inp, add_nvt, add_global_nose_hoover_chain
import torchfes as fes
from torchfes import md
from torchfes import properties as p
from torchfes.recorder.xyz import XYZRecorder


class R12(nn.Module):
    def forward(self, inp: Dict[str, Tensor]):
        pos = inp[p.pos]
        rr1 = pos[:, 0, :]
        rr2 = pos[:, 1, :]
        rr12 = rr2 - rr1
        r12 = rr12.norm(2, -1)[:, None]
        return r12


class Ang213(nn.Module):
    def forward(self, inp: Dict[str, Tensor]):
        pos = inp[p.pos]
        rr1 = pos[:, 0, :]
        rr2 = pos[:, 1, :]
        rr3 = pos[:, 2, :]
        rr12 = rr2 - rr1
        rr13 = rr3 - rr1
        r12 = rr12.norm(2, -1)
        r13 = rr13.norm(2, -1)
        cos = (rr12 * rr13).sum(-1) / (r12 * r13)
        return cos.acos()[:, None]


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

# import sys; sys.exit()


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
    # for v in vel[0] / Bohr * AUT:
    #     print('{} {} {}'.format(*v))
    mom = vel * mas[:, :, None]
    inp[p.mom] = mom
    return inp


def calc(eng, adj, inp):
    out = adj(inp)
    return eng(out)
    # return EvalEnergiesForces(eng)(
    #     inp, adj(pn.pnt_ful(inp[p.cel], inp[p.pbc], inp[p.pos], inp[p.ent])))


def main():
    parser = argparse.ArgumentParser('bme_lj')
    parser.add_argument('--dyn', type=str)
    parser.add_argument('--kbt', type=str, default='',)
    parser.add_argument('--colvar', type=str, default='')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    inp = make_inp()
    eng = EvalEnergies(
        LennardJones(e=torch.tensor([0.1]), s=torch.tensor([3.405]), rc=100.0)
    )
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(100.0), [(p.coo, 100.0)]
    )
    if args.colvar:
        if args.colvar == 'ang':
            col_var = Ang213()
            inp[p.bme_cen] = torch.ones([1, 1]) * pi / 2
        elif args.colvar == 'rad':
            col_var = R12()
            inp[p.bme_cen] = torch.ones([1, 1]) * 3.405
        else:
            raise KeyError()
    if args.kbt:
        if args.kbt == 'lan':
            kbt = md.unified.GlobalLangevin()
        elif args.kbt == 'nhc':
            kbt = md.unified.GlobalNHC(1)
        else:
            raise KeyError()
    dyn_key = args.dyn
    fire = fes.md.FIRE(0.5, 10, 0.5, 1.1, 0.5, 1.0 * fs)
    bme = False
    if dyn_key == 'PQ':
        dyn = md.PQ(eng, adj)
    elif dyn_key == 'PQP':
        dyn = md.PQP(eng, adj)
    elif dyn_key == 'PQF':
        dyn = md.PQF(eng, adj, fire)
    elif dyn_key == 'PQTQ':
        dyn = md.PQTQ(eng, adj, kbt)
    elif dyn_key == 'PQTQP':
        dyn = md.PQTQP(eng, adj, kbt)
    elif dyn_key == 'PTPQ':
        dyn = md.PTPQ(eng, adj, kbt)
    elif dyn_key == 'TPQPT':
        dyn = md.TPQPT(eng, adj, kbt)
    elif dyn_key == 'PTPQs':
        dyn = md.PTPQs(eng, adj, kbt, col_var, 1e-7, True)
        bme = True
    elif dyn_key == 'TPQsPTr':
        dyn = md.TPQsPTr(eng, adj, kbt, col_var, 1e-7, 1e-7, True)
        bme = True
    elif dyn_key == 'PQTQs':
        dyn = md.PQTQs(eng, adj, kbt, col_var, 1e-7, True)
        bme = True
    elif dyn_key == 'PQTQsPr':
        dyn = md.PQTQsPr(eng, adj, kbt, col_var, 1e-7, 1e-7, True)
        bme = True
    dyn = jit.script(dyn)

    xyz = XYZRecorder('xyz', 'w', ['Ar'], 1)
    timer = ignite.handlers.Timer()
    for _ in range(10):
        inp = dyn(inp)
        tim = inp[p.tim].item() / fs
        tim = round(decimal.Decimal(tim), 1)
        eng = round(decimal.Decimal(inp[p.eng].item() / Ha), 7)
        if bme:
            lmd = round(decimal.Decimal(inp[p.bme_lmd].item() / Ha), 7)
            print(f'{tim:>5} {eng:> 7} {lmd:> 7}')
        else:
            print(f'{tim:>5} {eng:> 7}')
        xyz.append(inp)
    print(timer.value())


if __name__ == "__main__":
    main()
