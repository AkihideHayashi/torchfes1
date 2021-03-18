from ase.io import read
import torchfes as fes
from ase.units import kB, fs
from torch import nn
from torch import Tensor
import os
from pathlib import Path
import pnpot
import pointneighbor as pn
from typing import Dict
import array
import torch
from torch import Tensor
from torchfes import properties as p


try:
    import plumed
except ModuleNotFoundError:
    pass


os.environ['PLUMED_KERNEL'] = '/Users/akihide/.local/lib/libplumedKernel.dylib'


def same(x: Tensor):
    v = x.mean().item()
    assert torch.all((x - v).abs() < 1e-8)
    return v


def _toarray(tensor):
    return array.array('d', tensor.detach().to('cpu').flatten().numpy())


class RPMDPlumed:
    def __init__(self, kbt, colvar, ndim, timestep, logpath='log.log'):
        # if 'plumed' not in locals():
        #     raise RuntimeError('plumed not imported.')
        self.ndim = ndim
        self.colvar = colvar
        self.box = array.array('d', [0] * (3 * 3))
        self.virial = array.array('d', [0] * (3 * 3))
        self.masses = array.array('d', [1] * ndim)
        self.charges = array.array('d', [0] * ndim)
        self.forces = array.array('d', [0] * 3 * ndim)
        self.positions = array.array('d', [0] * 3 * ndim)
        self.step = 0
        self.bias = array.array('d', [0])
        self.p = plumed.Plumed()
        print('timestep', timestep)
        self.p.cmd("setNatoms", ndim)
        self.p.cmd("setKbT", kbt)
        self.p.cmd("setTimestep", timestep)
        self.p.cmd("setLogFile", logpath)
        self.p.cmd("init")
        self.commands = []

    def calc(self):
        self.p.cmd("setStep", self.step)
        self.p.cmd("setBox", self.box)
        self.p.cmd("setMasses", self.masses)
        self.p.cmd("setCharges", self.charges)
        self.p.cmd("setPositions", self.positions)
        self.p.cmd("setForces", self.forces)
        self.p.cmd("setVirial", self.virial)
        self.p.cmd("calc")

    def readInputLine(self, command):
        self.p.cmd("readInputLine", command)
        self.commands.append(command)

    def read(self, label, command, **args):
        line = make_plumed_command(label, command, **args)
        self.readInputLine(line)

    def __call__(self, mol: Dict[str, Tensor]):
        if not mol[fes.p.pos].requires_grad:
            mol[fes.p.pos].requires_grad_()
        colvar = self.colvar(mol).mean(0)
        col = torch.zeros([colvar.size(0), 3])
        col[:, 0] = colvar
        self.step = same(mol[fes.p.stp])
        self.positions[:] = _toarray(col)
        self.forces[:] = array.array('d', [0.0] * self.ndim * 3)
        self.virial[:] = array.array('d', [0.0] * 9)
        self.bias[0] = 0.0
        self.calc()
        self.p.cmd("getBias", self.bias)
        forces = torch.tensor(self.forces).view_as(col)
        frc = forces[:, 0]
        g, = torch.autograd.grad(colvar, mol[fes.p.pos], -frc)
        f = -g
        ret = mol.copy()
        ret[p.eng_res] += self.bias[0]
        ret[p.eng] += self.bias[0]
        ret[p.frc][:] += f
        return ret


class MyColvar(nn.Module):
    def forward(self, mol):
        return mol[fes.p.pos][:, 0, 0:1]


kbt = 0.1
beta = 10.0
n_ring = 16
hbar = 1.0
k = 1.0
kbt = 0.1
beta = 10.0


def make_inp():
    n_atm = 1
    n_bch = n_ring
    n_dim = 1
    cel = torch.eye(n_dim)[None, :, :].expand((n_bch, n_dim, n_dim)) * 1000
    pos = torch.rand([n_bch, n_atm, n_dim])
    pbc = torch.zeros([n_bch, n_dim], dtype=torch.bool)
    elm = torch.ones([n_bch, n_atm])
    mas = torch.ones([n_bch, n_atm])
    inp = fes.mol.init_mol(cel, pbc, elm, pos, mas)
    inp = fes.mol.add_nvt(inp, 0.01, 0.1)
    inp = fes.mol.add_global_langevin(inp, 0.7)
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
    mol[fes.p.dtm] = mol[fes.p.dtm].mean(dim=0, keepdim=True)
    mdl = pnpot.classical.Quadratic(torch.tensor([k]))
    adj = fes.adj.SetAdjSftSpcVecSod(
        pn.Coo2FulSimple(1.0), [(fes.p.coo, 1.0)]
    )
    evl = fes.ff.get_adj_eng_frc(adj, mdl)
    mom = fes.md.UpdtMom(0.5)
    base = fes.md.rpmd_base(hbar, beta, n_ring, 'cpu')
    pos = fes.md.RpmdPos(base, mol[fes.p.mas], mol[fes.p.dtm].item(), 1)
    lan = fes.md.RpmdLangevin(
        base, 0.7, mol[fes.p.dtm].item() * 0.5, mol[fes.p.mas], 1)
    kin = fes.md.RpmdKin(base)
    # mol[fes.p.mom][:, 0, 0] = torch.tensor([-0.39253548, -0.23131893, -0.39253548, -0.23131893])
    mol = evl(mol)
    pos_lst = []
    pot_lst = []
    kin_lst = []
    mol[fes.p.mom][:, 0, 0] = torch.ones(n_ring)
    mol[fes.p.pos][:, 0, 0] = torch.zeros(n_ring)
    colvar = MyColvar()
    plumed_ = RPMDPlumed(kbt, colvar, 1, 0.5 * fs)
    plumed_.readInputLine('a: POSITION ATOM=1 NOPBC')
    plumed_.readInputLine('metad: METAD ARG=a.x SIGMA=0.1 HEIGHT=0.1 PACE=1 CALC_RCT GRID_MIN=-10.0 GRID_MAX=10.0 BIASFACTOR=10.0')
    plumed_.readInputLine('PRINT ARG=a.x,metad.* FILE=COLVAR')
    for i in range(40000):
        mol = lan(mol)
        mol = mom(mol)
        mol = pos(mol)
        mol = evl(mol)
        mol = plumed_(mol)
        mol = kin(mol)
        mol = mom(mol)
        mol = lan(mol)
        # print(mol[fes.p.mom].flatten())
        # print(mol[fes.p.pos].flatten())
        pos_lst.append(mol[fes.p.pos][:, 0, 0].detach())
        pot_lst.append(mol[fes.p.eng][:].detach())
        kin_lst.append(mol[fes.p.rpm_kin].detach())
        print(i)
    with open('tmp.pkl', 'wb') as f:
        torch.save([pos_lst, pot_lst, kin_lst], f)


main()
