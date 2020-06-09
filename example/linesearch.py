from typing import List, NamedTuple, Dict
import torch
from torch import Tensor, nn
from pnpot.classical.quad import Quadratic
from torchfes.inp import init_inp, init_nvt
from torchfes.forcefield import EvalEnergiesForcesGeneral
from torchfes.opt.linesearch import WolfeCondition, LogSmapler
from torchfes.opt.cg import CG, FRCG
from torchfes.general import cartesian_coordinate, CartesianCoordinate
from torchfes.general import PosEngFrc
from torchfes import properties as p


def make_inp():
    cel = torch.eye(3)[None, :, :] * 100.0
    pbc = torch.tensor([False, False, False])[None, :]
    pos = torch.tensor([[[1.0, 1.0, 1.0]], [[0.0, 1.0, 1.0]]])
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


def main_double_loop():
    torch.set_default_dtype(torch.float64)

    evl = EvalEnergiesForcesGeneral(
        Quadratic(torch.tensor([10.0, 1.0, 0.1])), [], CartesianCoordinate())
    env = make_inp()
    pos = cartesian_coordinate(env)
    stp_ = LogSmapler(0.5)
    con_ = WolfeCondition(0.4, 0.6)
    vec_ = CG(evl, FRCG())
    pef: PosEngFrc
    # init
    vec, pef = vec_.init(pos, env)
    for _ in range(10):
        stp = stp_.init(pef, pef.eng == pef.eng, False)
        while True:
            pos_tmp = pef.pos + stp * vec
            pef_tmp = evl(env, pos_tmp)
            con = con_(pef, pef_tmp, stp, vec)
            if (con == 0).all():
                break
            else:
                stp = stp_(con, pef_tmp)
        pef = pef_tmp
        vec = vec_(pef, env)
        print(pef.eng.squeeze(-1))


if __name__ == "__main__":
    main_double_loop()
