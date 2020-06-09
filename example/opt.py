import torch
from torch import jit
from pnpot.classical.quad import Quadratic
from torchfes.inp import init_inp, init_nvt
from torchfes.forcefield import EvalEnergiesForces
from torchfes.opt.linesearch import LineSearchOptimizer, LogSmapler, WolfeCondition
from torchfes.opt.cg import HSCG, PRCG, PRPCG, FRCG, CG
from torchfes.opt.newtoncg import NewtonCG, LineSearchNewtonCG
from torchfes.opt.lbfgs import LBFGS
from torchfes.md.md import PQF
from torchfes import properties as p


def make_inp():
    cel = torch.eye(3)[None, :, :] * 100.0
    pbc = torch.tensor([False, False, False])[None, :]
    pos = torch.tensor([[[0.0, 1.0, 1.0]]])
    elm = torch.tensor([0])[None, :]
    inp = init_inp(cel, pbc, elm, pos)
    init_nvt(inp, mas=torch.tensor([1.0]), dtm=torch.tensor([0.1]), kbt=torch.tensor([1.0]))
    inp[p.mom] *= 0.0
    for key in inp:
        size = list(inp[key].size())
        size[0] = 2
        inp[key] = inp[key].expand(size).clone()
    inp[p.pos][1] = torch.tensor([1.0, 0.0, 0.0])
    return inp


def main1():
    torch.set_default_dtype(torch.float64)

    evl = EvalEnergiesForces(Quadratic(torch.tensor([10.0, 1.0, 0.1])), [])
    inp = make_inp()
    inp = evl(inp)
    # opt = LineSearchOptimizer(evl, CG(PRPCG()), sampler=LogSmapler(0.1, 0.5), condition=WolfeCondition(0.4, 0.6))
    # opt = LineSearchNewtonCG(evl, False)
    # opt = PQF(evl, a0=1.0, n_min=1, f_a=0.9, f_inc=1.1, f_dec=0.9, dtm_max=0.1)
    opt = LineSearchOptimizer(evl, LBFGS(0.1))
    print(inp[p.pos])
    for _ in range(100):
        inp = opt(inp)
        print('pos: ', inp[p.pos])#, 'eng: ', inp[p.eng_tot])


if __name__ == "__main__":
    main1()
