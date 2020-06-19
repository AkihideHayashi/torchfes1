import argparse

import torch
from torch import jit

import pointneighbor as pn
from pnpot.classical.quad import Quadratic
from torchfes import properties as p
from torchfes.forcefield import EvalEnergiesForcesGeneral, EvalEnergies
from torchfes.inp import init_inp, add_nvt
from torchfes import opt as fopt
from torchfes.general import cartesian_coordinate, CartesianCoordinate


def make_inp():
    cel = torch.eye(3)[None, :, :] * 100.0
    pbc = torch.tensor([False, False, False])[None, :]
    pos = torch.tensor([[[0.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]])
    elm = torch.tensor([0])[None, :]
    inp = init_inp(cel, pbc, elm, pos, mas=torch.tensor([1.0])[elm])
    add_nvt(inp, dtm=torch.tensor([0.1]), kbt=torch.tensor([0.0]))
    for key in inp:
        size = list(inp[key].size())
        size[0] = 2
        inp[key] = inp[key].expand(size).clone()
    return inp


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Linesearch optimizer test on quadratic function',
    )
    parser.add_argument('method', type=str)
    args = parser.parse_args()
    return args.method


def main():
    method = parse_args()
    torch.set_default_dtype(torch.float64)

    gen = CartesianCoordinate()
    eng = EvalEnergies(Quadratic(torch.tensor([10.0, 1.0, 0.1])))
    adj = pn.Coo2FulSimple(10.0)
    evl_gen = EvalEnergiesForcesGeneral(eng, gen, adj)
    env = make_inp()
    sync = False
    jitable = True
    if method == 'cg':
        vec = fopt.CG(evl_gen, fopt.FRCG())
    elif method == 'newton-cg':
        vec = fopt.NewtonCG(evl_gen)
    elif method == 'bfgs-diag':
        vec = fopt.BFGS(evl_gen, fopt.QuasiNewtonInitWithDiagonal(1.0))
    elif method == 'bfgs-exact':
        vec = fopt.BFGS(evl_gen, fopt.QuasiNewtonInitWithExact(evl_gen, 1e-2))
    elif method == 'lbfgs':
        vec = fopt.LBFGS(evl_gen, 10, 1.0)
        sync = True
        jitable = False
    else:
        raise NotImplementedError(method)
    stp = fopt.LogSmapler(2.0, 0.9, 1.0)
    con = fopt.WolfeCondition(0.4, 0.6, 1e-4, 1e-4)
    opt = fopt.GeneralLineSearchOptimizer(
        evl_gen, vec, stp, con, reset=False, sync=sync)
    pos = cartesian_coordinate(env)
    opt.init(env, pos)
    if jitable:
        opt = jit.script(opt)
    print(env[p.pos])
    for _ in range(60):
        env = opt(env)
        print('eng: ', env[p.eng].squeeze(-1))


if __name__ == "__main__":
    main()
