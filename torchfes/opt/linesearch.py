from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p
from ..utils import grad


def limit_step_size(stp: Tensor, siz: float):
    # stp_siz = stp.norm(p=2, dim=1)[:, None].expand_as(stp)
    stp_siz, _ = stp.max(dim=1)
    stp = torch.where(
        stp_siz > siz,
        stp / stp_siz * siz,
        stp
    )
    return stp


class LimitStepSize(nn.Module):
    def __init__(self, stp, max_stp: float):
        super().__init__()
        self.stp = stp  # Newton direction.
        self.max_stp = max_stp

    def forward(self, inp: Dict[str, Tensor]):
        out = self.stp(inp)
        stp = out[p.gen_dir]
        stp = limit_step_size(stp, self.max_stp)
        out[p.gen_vec] = stp
        return out

from .hessian import ExactHessian
from .newton import NewtonSolve

def newton(eng, gen, spe, mol):
    hes = ExactHessian()
    new = NewtonSolve(hes)
    mol = gen(mol)
    mol = eng(mol)
    mol = hes(mol)
    while True:
        mol = new(mol)
        mol[p.gen_stp] = torch.zeros_like(
            mol[p.gen_eng], requires_grad=True)[:, None]
        gen_dir_nrm = mol[p.gen_dir] / mol[p.gen_dir].norm(p=1, dim=1)[:, None]
        gen_stp_grd_ini = grad(mol[p.gen_eng], mol[p.gen_stp])
        while True:
            mol[p.gen_vec] = mol[p.gen_dir] * mol[p.gen_stp]
            mol[p.gen_pos] = mol[p.gen_pos] + mol[p.gen_vec]
            mol = spe(mol)
            mol = gen(mol)
            mol = eng(mol)
            gen_stp_grd = grad(mol[p.gen_eng], mol[p.gen_stp], create_graph=True)
            if (gen_stp_grd.abs() < gen_stp_grd_ini.abs() * 0.5).all():
                break
            print(gen_stp_grd.size())
            print(gen_dir_nrm.size())
            gen_stp_hes = grad(mol[p.gen_grd], mol[p.gen_stp], gen_dir_nrm)
            stp = - gen_stp_grd / gen_stp_hes
            mol[p.gen_stp] += stp
            print(mol[p.gen_stp])
        mol[p.gen_vec] = mol[p.gen_dir] * mol[p.gen_stp]
        mol[p.gen_pos] = mol[p.gen_pos] + mol[p.gen_vec]
        mol = spe(mol)
        mol = gen(mol)
        mol = eng(mol)
        mol = hes(mol)
        print(mol[p.gen_grd].abs().max())
        if mol[p.gen_grd].abs().max() < 1e-4:
            break
