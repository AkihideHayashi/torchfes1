from typing import Dict, Optional

from torch import Tensor, nn

from . import properties as p
from .api import Energies
from .general import PosEngFrc
from .utils import detach_, grad
from .adj import SetAdjSftSpcVecSod


class EvalEnergies(nn.Module):
    def __init__(self, mdl, res: Optional[nn.Module] = None):
        super().__init__()
        self.mdl = mdl
        self.res = res

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        eng_mdl: Energies = self.mdl(out)
        out[p.eng_atm] = eng_mdl.eng_atm
        out[p.eng_mol] = eng_mdl.eng_mol
        out[p.eng_atm_std] = eng_mdl.eng_atm_std
        out[p.eng_mol_std] = eng_mdl.eng_mol_std
        out[p.eng_atm_ens] = eng_mdl.eng_atm_ens
        out[p.eng_mol_ens] = eng_mdl.eng_mol_ens
        n_bch = out[p.pos].size(0)
        if self.res is not None:
            eng_res = self.res(out)
        else:
            eng_res = eng_mdl.eng_mol.new_zeros([n_bch, 0])
        eng_tot = eng_mdl.eng_mol + eng_res.sum(1)
        out[p.eng_res] = eng_res
        out[p.eng] = eng_tot
        return out


class EvalEnergiesForces(nn.Module):
    def __init__(self, eng: EvalEnergies, mod: Optional[nn.Module] = None):
        super().__init__()
        assert isinstance(eng, EvalEnergies)
        self.eng = eng
        self.mod = mod
        if (self.mod is not None and hasattr(self.mod, 'create_graph')
                and self.mod.create_graph):
            self.create_graph = True
        else:
            self.create_graph = False

    def forward(self, inp: Dict[str, Tensor],
                requires_frc: bool = True,
                requires_sts: bool = False,
                create_graph: bool = False,
                retain_graph: Optional[bool] = None):
        if self.create_graph:
            create_graph = True
        pos = inp[p.pos]
        cel = inp[p.cel]
        if requires_frc and not pos.requires_grad:
            pos.requires_grad_()
        if requires_sts and not cel.requires_grad:
            cel.requires_grad_()
        out: Dict[str, Tensor] = self.eng(inp)
        eng_mol = out[p.eng_mol]
        eng_res = out[p.eng_res].sum(1)
        assert eng_mol.dim() == 1
        assert eng_res.dim() == 1
        if requires_frc:
            out[p.frc_mol] = grad(-eng_mol, pos, create_graph=create_graph,
                                  retain_graph=True)
            out[p.frc_res] = grad(-eng_res, pos, create_graph=create_graph,
                                  retain_graph=True)
            out[p.frc] = out[p.frc_mol] + out[p.frc_res]
        if requires_sts:
            out[p.sts_mol] = grad(-eng_mol, cel, create_graph=create_graph,
                                  retain_graph=True)
            out[p.sts_res] = grad(-eng_res, pos, create_graph=create_graph,
                                  retain_graph=True)
            out[p.sts] = out[p.sts_mol] + out[p.sts_res]
        if self.mod is not None:
            out = self.mod(out, create_graph=create_graph, retain_graph=True)
        if p.fix_msk in out:
            out[p.frc].masked_fill_(out[p.fix_msk].squeeze(0), 0.0)
        if retain_graph is None:
            retain_graph = create_graph
        if not retain_graph:
            detach_(out)
        return out


class EvlAdjEngFrc(nn.Module):
    def __init__(self, adj, evl):
        super().__init__()
        self.adj = adj
        self.evl = evl

    def forward(self, mol: Dict[str, Tensor],
                requires_frc: bool = True,
                requires_sts: bool = False,
                create_graph: bool = False,
                retain_graph: Optional[bool] = None):
        return self.evl(self.adj(mol), requires_frc, requires_sts,
                        create_graph, retain_graph)


def get_adj_eng_frc(adj: SetAdjSftSpcVecSod, mdl: nn.Module,
                    ext_eng: Optional[nn.Module] = None,
                    ext_frc: Optional[nn.Module] = None):
    eng = EvalEnergies(mdl, ext_eng)
    frc = EvalEnergiesForces(eng, ext_frc)
    ret = EvlAdjEngFrc(adj, frc)
    return ret


class EvalEnergiesForcesGeneral(nn.Module):

    def __init__(self, eng: EvalEnergies, gen: nn.Module, adj: nn.Module):
        super().__init__()
        self.eng = EvalEnergiesForces(eng)
        self.gen = gen
        self.adj = adj

    def forward(self, env: Dict[str, Tensor], pos: Tensor,
                create_graph: bool = False):
        if not pos.requires_grad:
            pos = pos.clone().requires_grad_()
        inp = self.gen(env, pos)
        out = self.adj(inp)
        out = self.eng(out, retain_graph=True)
        eng_tot = out[p.eng]
        frc = grad(-eng_tot, pos, create_graph=create_graph)
        if not create_graph:
            detach_(out)
            pef = PosEngFrc(pos=pos.clone().detach(),
                            eng=eng_tot.clone().detach().unsqueeze(-1),
                            frc=frc.clone().detach())
            return out, pef
        else:
            pef = PosEngFrc(pos=pos, eng=eng_tot.unsqueeze(-1), frc=frc)
            return out, pef
