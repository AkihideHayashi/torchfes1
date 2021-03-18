from typing import Dict, Optional
import torch
from torch import nn, Tensor
from ... import properties as p
from ...forcefield import EvlAdjEngFrc
from ...utils import grad


def generalize(mol: Dict[str, Tensor]):
    mol = mol.copy()
    if p.fix_msk in mol:
        gen = mol[p.pos][:, ~mol[p.fix_msk].squeeze(0)]
    else:
        gen = mol[p.pos].flatten(1)
    if p.con_mul in mol:
        gen = torch.cat([mol[p.con_mul], gen], dim=1)
    gen.requires_grad_()
    mol[p.gen_pos] = gen
    return mol


class Lagrangian(nn.Module):
    def __init__(self, evl: EvlAdjEngFrc, con: Optional[nn.Module] = None):
        super().__init__()
        self.evl = evl
        self.con = con

    def forward(self, mol: Dict[str, Tensor], create_graph: bool = False):
        mol = mol.copy()
        gen = mol[p.gen_pos].clone().detach().requires_grad_()
        mol[p.gen_pos] = gen
        if self.con is not None:
            _, num_con = mol[p.con_mul].size()
            mol[p.con_mul] = gen[:, :num_con]
            gen = gen[:, num_con:]
        if p.fix_msk in mol:
            mol[p.pos] = mol[p.pos].masked_scatter(~mol[p.fix_msk].squeeze(0),
                                                   gen)
        else:
            mol[p.pos] = gen.view_as(mol[p.pos])
        mol = self.evl(mol, create_graph=create_graph)
        if self.con is not None:
            con = self.con(mol) - mol[p.con_cen]  # bch, con
            frc = mol[p.frc] + grad((mol[p.con_mul] * con).sum(1), mol[p.pos],
                                    create_graph=create_graph, retain_graph=True)
        else:
            frc = mol[p.frc]
        if p.fix_msk in mol:
            gen_grd = -frc[:, ~mol[p.fix_msk].squeeze(0)]
        else:
            gen_grd = -frc.flatten(1)
        if self.con is not None:
            con = self.con(mol) - mol[p.con_cen]  # bch, con
            gen_grd = torch.cat([-con, gen_grd], dim=1)
        mol[p.gen_grd] = gen_grd
        return mol
