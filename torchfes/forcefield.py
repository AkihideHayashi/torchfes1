from typing import Dict, Optional

from torch import Tensor, nn

from pointneighbor import AdjSftSpc

from . import properties as p
from .api import Energies
from .general import PosEngFrc
from .utils import detach_, grad, pnt_ful


class EvalEnergies(nn.Module):
    def __init__(self, mdl, res: Optional[nn.Module] = None):
        super().__init__()
        self.mdl = mdl
        self.res = res

    def forward(self, inp: Dict[str, Tensor], adj: AdjSftSpc):
        eng_mdl: Energies = self.mdl(inp, adj)
        n_bch = inp[p.pos].size(0)
        if self.res is not None:
            eng_res = self.res(inp, adj)
        else:
            eng_res = eng_mdl.eng_mol.new_zeros([n_bch, 0])
        eng_tot = eng_mdl.eng_mol + eng_res.sum(1)
        out = inp.copy()
        out[p.eng_atm] = eng_mdl.eng_atm
        out[p.eng_mol] = eng_mdl.eng_mol
        out[p.eng_atm_std] = eng_mdl.eng_atm_std
        out[p.eng_mol_std] = eng_mdl.eng_mol_std
        out[p.eng_res] = eng_res
        out[p.eng] = eng_tot
        return out


class EvalEnergiesForces(nn.Module):
    def __init__(self, eng: nn.Module):
        super().__init__()
        self.eng = eng

    def forward(self, inp: Dict[str, Tensor], adj: AdjSftSpc,
                frc_pos: bool = True, frc_cel: bool = False,
                frc_grd: bool = False, retain_graph: Optional[bool] = None):
        pos = inp[p.pos]
        cel = inp[p.cel]
        if frc_pos and not pos.requires_grad:
            pos.requires_grad_()
        if frc_cel and not cel.requires_grad:
            cel.requires_grad_()
        out: Dict[str, Tensor] = self.eng(inp, adj)
        eng_mol = out[p.eng_mol]
        eng_res = out[p.eng_res].sum(1)
        assert eng_mol.dim() == 1
        assert eng_res.dim() == 1
        if frc_pos:
            out[p.frc_mol] = grad(-eng_mol, pos, create_graph=frc_grd,
                                  retain_graph=retain_graph)
            out[p.frc_res] = grad(-eng_res, pos, create_graph=frc_grd,
                                  retain_graph=retain_graph)
            out[p.frc] = out[p.frc_mol] + out[p.frc_res]
        if frc_cel:
            out[p.sts_mol] = grad(-eng_mol, cel, create_graph=frc_grd,
                                  retain_graph=retain_graph)
            out[p.sts_res] = grad(-eng_res, pos, create_graph=frc_grd,
                                  retain_graph=retain_graph)
            out[p.sts] = out[p.sts_mol] + out[p.sts_res]
        if retain_graph is None:
            retain_graph = frc_grd
        if not retain_graph:
            detach_(out)
        return out


class EvalEnergiesForcesGeneral(nn.Module):

    def __init__(self, eng: nn.Module, gen: nn.Module, adj: nn.Module):
        super().__init__()
        self.eng = EvalEnergiesForces(eng)
        self.gen = gen
        self.adj = adj

    def forward(self, env: Dict[str, Tensor], pos: Tensor,
                frc_grd: bool = False):
        if not pos.requires_grad:
            pos = pos.clone().requires_grad_()
        inp = self.gen(env, pos)
        adj = self.adj(pnt_ful(inp))
        out = self.eng(inp, adj, retain_graph=True)
        eng_tot = out[p.eng]
        frc = grad(-eng_tot, pos, create_graph=frc_grd)
        if not frc_grd:
            detach_(out)
            pef = PosEngFrc(pos=pos.clone().detach(),
                            eng=eng_tot.clone().detach().unsqueeze(-1),
                            frc=frc.clone().detach())
            return out, pef
        else:
            pef = PosEngFrc(pos=pos, eng=eng_tot.unsqueeze(-1), frc=frc)
            return out, pef
