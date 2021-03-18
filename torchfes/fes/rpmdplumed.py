from typing import Dict
import array
import torch
from torch import Tensor
from torchfes import properties as p

try:
    import plumed
except ModuleNotFoundError:
    pass


def _toarray(tensor):
    return array.array('d', tensor.detach().to('cpu').numpy())


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
        # print(line)
        self.readInputLine(line)

    def __call__(self, mol: Dict[str, Tensor]):
        assert mol[p.stp].size(0) == 1, mol[p.stp].size()
        colvar = self.colvar(mol).mean(0)
        self.step = int(mol[p.stp].item())
        self.positions[:] = _toarray(colvar)
        self.forces[:] = array.array('d', [0.0] * self.ndim * 3)
        self.virial[:] = array.array('d', [0.0] * 9)
        self.calc()
        self.p.cmd("getBias", self.bias)
        torch.tensor(self.forces).reshape
        ret = mol.copy()
        ret[p.eng_res] += self.bias[0]
        ret[p.eng] += self.bias[0]
        ret[p.frc][:] = torch.tensor(self.forces).reshape_as(ret[p.frc])
        return ret


def make_plumed_command(label, command, **args):
    def _inner(key, val):
        if isinstance(val, torch.Tensor):
            val = val.tolist()
        if isinstance(val, list):
            v = ','.join(map(str, val))
            return f'{key}={v}'
        if isinstance(val, str):
            return f'{key}={val}'
        if isinstance(val, (int, float)):
            return f'{key}={val}'
        if val is None:
            return f'{key}'
        else:
            raise NotImplementedError()
    arg = ' '.join(_inner(key, val) for key, val in args.items())
    return f'{label}: {command} {arg}'
