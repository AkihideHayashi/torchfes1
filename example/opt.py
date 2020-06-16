import torch
from torch import jit

from pnpot.classical.quad import Quadratic
from torchfes import properties as p
from torchfes.forcefield import EvalEnergiesForcesGeneral
from torchfes.inp import init_inp, init_nvt
# from torchfes.md.md import PQF
from torchfes import opt as fopt
from torchfes.general import cartesian_coordinate, CartesianCoordinate


def make_inp():
    cel = torch.eye(3)[None, :, :] * 100.0
    pbc = torch.tensor([False, False, False])[None, :]
    pos = torch.tensor([[[0.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]])
    elm = torch.tensor([0])[None, :]
    inp = init_inp(cel, pbc, elm, pos)
    init_nvt(inp, mas=torch.tensor([1.0]), dtm=torch.tensor([0.1]),
             kbt=torch.tensor([0.0]))
    inp[p.mom] *= 0.0
    for key in inp:
        size = list(inp[key].size())
        size[0] = 2
        inp[key] = inp[key].expand(size).clone()
    return inp


def main1():
    torch.set_default_dtype(torch.float64)

    gen = CartesianCoordinate()
    evl = EvalEnergiesForcesGeneral(
        Quadratic(torch.tensor([10.0, 1.0, 0.1])), [], gen)
    env = make_inp()
    # vec = fopt.CG(evl, fopt.FRCG())
    # vec = fopt.NewtonCG(evl)
    # vec = fopt.BFGS(evl, fopt.QuasiNewtonInitWithDiagonal(0.01))
    # vec = fopt.BFGS(evl, fopt.QuasiNewtonInitWithExact(evl))
    vec = fopt.LBFGS(evl, 10, 0.01)
    opt = fopt.LineSearchOptimizer(
        evl, vec, fopt.LogSmapler(2.0, 0.9), fopt.WolfeCondition(0.4, 0.6),
        reset=True, sync=True)
    pos = cartesian_coordinate(env)
    opt.init(env, pos)
    # opt = jit.script(opt)
    print(env[p.pos])
    for _ in range(60):
        pef = opt(env)
        print('eng: ', pef.eng.squeeze(-1))


if __name__ == "__main__":
    main1()
