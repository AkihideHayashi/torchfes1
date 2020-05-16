from typing import Dict, List
import torch
from torch import nn, Tensor
from . import properties as p
from .api import Energies


class EvalEnergies(nn.Module):
    def __init__(self, mdl, res: List[nn.Module]):
        super().__init__()
        self.mdl = mdl
        self.res = nn.ModuleList(res)

    def forward(self, inp: Dict[str, Tensor]):
        eng_mdl: Energies = self.mdl(inp)
        eng_tot = eng_mdl.eng_mol
        eng_res = torch.zeros_like(eng_tot)
        for res in self.res:
            eng_res = eng_res + res(inp)
        eng_tot = eng_tot + eng_res
        out = inp.copy()
        out[p.eng_atm] = eng_mdl.eng_atm
        out[p.eng_mol] = eng_mdl.eng_mol
        out[p.eng_atm_std] = eng_mdl.eng_atm_std
        out[p.eng_mol_std] = eng_mdl.eng_mol_std
        out[p.eng_res] = eng_res
        out[p.eng_tot] = eng_tot
        return out


class EvalForcesOnly(nn.Module):
    def __init__(self, evl):
        super().__init__()
        self.evl = evl

    def forward(self, inp: Dict[str, Tensor]):
        pos = inp[p.pos].clone().detach().requires_grad_(True)
        out = inp.copy()
        out[p.pos] = pos
        out = self.evl(out)
        eng_tot = out[p.eng_tot]
        frc, = torch.autograd.grad(-eng_tot, pos, torch.ones_like(eng_tot))
        out[p.frc] = frc
        return {key: val.detach() for key, val in out.items()}
