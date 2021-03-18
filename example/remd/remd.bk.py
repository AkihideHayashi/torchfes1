from random import random, choice
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
    inp = fes.mol.init_mol(cel, pbc, elm, pos, mas)
    inp = fes.mol.add_nvt(inp, 1.0 * fs, 300 * kB)
    inp = fes.mol.add_global_langevin(inp, 100.0 * fs)
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
    kbt = reprex_kbt(30, 100.0, 500.0)
    mol = fes.data.reprecate(mol, len(kbt))
    mol[fes.p.idt] = torch.arange(len(kbt))
    mol[fes.p.kbt] = kbt
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    evl = fes.ff.get_adj_eng_frc(adj, mdl)
    kbt = fes.md.GlobalLangevin()
    dyn = fes.md.leap_frog(evl, kbt=kbt)
    timer = ignite.handlers.Timer()
    mol = evl(mol)
    rex_stp = 20
    num_accept = torch.tensor([0 for _ in range(len(mol[fes.p.kbt]))], dtype=torch.float32)
    num_reject = torch.tensor([0 for _ in range(len(mol[fes.p.kbt]))], dtype=torch.float32)
    with fes.rec.open_trj(trj_path, mode) as rec:
        for i in range(100000):
            mol = dyn(mol)
            rec.put(mol)
            if i > 1000 and i % rex_stp == 0:
                (idi, idj), changed = replica_exchange(mol)
                # print(idi, idj, changed)
                if changed:
                    num_accept[min(idi, idj)] += 1
                else:
                    num_reject[min(idi, idj)] += 1
                num = num_accept + num_reject
                print(num, changed)
                if (num[:-1] > 10).all():
                    mol[fes.p.kbt] = better_kbt(num_accept, num_reject, mol[fes.p.kbt])
                    num_accept[:] = 0.0
                    num_reject[:] = 0.0



def swap(x, i, j):
    tmp = x[i].clone()
    x[i] = x[j]
    x[j] = tmp


def get_index(idt, i):
    return idt.tolist().index(i)

def is_sorted(x):
    return (x.sort()[0] == x).all()

def replica_exchange(mol):
    eng_pot = mol[fes.p.eng]
    eng_kin = fes.fn.kinetic_energies(
        mol[fes.p.mom], mol[fes.p.mas], mol[fes.p.ent])
    eng = eng_pot + eng_kin
    idt = mol[fes.p.idt]
    kbt = mol[fes.p.kbt]
    idx = kbt.argsort()
    idi = choice(idx).item()
    # assert is_sorted(kbt[idx])
    # assert is_sorted(kbt[idx])
    if random() > 0.5:
        idj = idi + 1
    else:
        idj = idi - 1
    if idi == 0:
        idj = 1
    if idi == idt.max().item():
        idj = idi - 1
    i = get_index(idt, idi)
    j = get_index(idt, idj)
    bi = 1 / kbt[i]
    bj = 1 / kbt[j]
    ei = eng[i]
    ej = eng[j]
    delta = (bi - bj) * (ei - ej)
    p = torch.exp(delta).item()
    p = min(1, p)
    xi = random()
    if xi < p:
        ti = 1 / bi
        tj = 1 / bj
        mol[fes.p.mom][i, :, :] *= torch.sqrt(tj / ti)
        mol[fes.p.mom][j, :, :] *= torch.sqrt(ti / tj)
        swap(mol[fes.p.idt], i, j)
        swap(mol[fes.p.kbt], i, j)
        # assert (torch.argsort(mol[fes.p.idt]) == torch.argsort(mol[fes.p.kbt])).all()
        mol[fes.p.idt].detach_()
        mol[fes.p.kbt].detach_()
        return (idi, idj), True
    else:
        return (idi, idj), False


def reprex_kbt(kbt0, d0, kbtn):
    b0 = torch.tensor(1 / (kbt0 * kB))
    d0 = torch.tensor(1 / (kbt0 * kB)) - torch.tensor(1 / ((kbt0 + d0) * kB))
    bn = torch.tensor(1 / (kbtn * kB))
    n = (torch.log(bn) - torch.log(b0)) / (torch.log(b0 + d0) - torch.log(b0))
    n = -n
    n = torch.arange(1, int(n.item()) + 1)
    bn = torch.exp(n * torch.log(b0 - d0) - (n - 1) * torch.log(b0))
    bn = torch.tensor([b0] + bn.tolist())
    return 1 / (bn * kB)

def better_kbt(accept, reject, kbt_):
    kbt, idx = torch.sort(kbt_)
    p = (accept[:-1] / (accept[:-1] + reject[:-1]))
    beta = 1 / kbt
    delta = beta[1:] - beta[:-1]
    beta = beta[:-1]
    ln02 = torch.log(torch.tensor(0.2))
    lnp = torch.log(p.to(kbt))
    sqrtlnln = torch.sqrt(ln02 / lnp)
    c = 1 + delta / beta * sqrtlnln
    new_beta = kbt.clone()
    new_beta[0] = beta[0]
    for m in range(len(beta) - 1):
        new_beta[m + 1] = new_beta[m] * c[m]
    return 1 / new_beta



if __name__ == '__main__':
    main()

