from typing import Dict, List, Tuple
from torch import Tensor, nn
import pointneighbor as pn
from pointneighbor.neighbor.lil2.lil2 import (
    transform_tensor_coo_to_lil, transformation_mask_coo_to_lil
)
from pointneighbor import AdjSftSpc, VecSod
from .utils import pnt_ful
from . import properties as p


def get_adj_sft_spc(inp: Dict[str, Tensor], typ: str, rc: float):
    adj = inp[p.nei_adj(typ, rc)]
    sft = inp[p.nei_sft(typ, rc)]
    spc = inp[p.nei_spc(typ, rc)]
    return AdjSftSpc(adj=adj, sft=sft, spc=spc)


def get_vec_sod(inp: Dict[str, Tensor], typ: str, rc: float):
    vec = inp[p.vec(typ, rc)]
    sod = inp[p.sod(typ, rc)]
    return VecSod(vec=vec, sod=sod)


def set_coo_adj_sft_spc_vec_sod(
        inp: Dict[str, Tensor], adj_sft_spc: AdjSftSpc, vec_sod: VecSod,
        rc: float, set_vec_sod: bool):
    adj_out, vec_out = pn.cutoff_coo2(adj_sft_spc, vec_sod, rc)
    out = inp.copy()
    out[p.nei_adj(p.coo, rc)] = adj_out.adj
    out[p.nei_sft(p.coo, rc)] = adj_out.sft
    out[p.nei_spc(p.coo, rc)] = adj_out.spc
    if set_vec_sod:
        out[p.vec(p.coo, rc)] = vec_out.vec
        out[p.sod(p.coo, rc)] = vec_out.sod
    return out


def set_lil_adj_sft_spc_vec_sod(
        inp: Dict[str, Tensor], rc: float, set_vec_sod: bool):
    out = inp.copy()
    coo = AdjSftSpc(
        adj=inp[p.nei_adj(p.coo, rc)],
        sft=inp[p.nei_sft(p.coo, rc)],
        spc=inp[p.nei_spc(p.coo, rc)],
    )
    lil = pn.coo_to_lil(coo)
    out[p.nei_adj(p.lil, rc)] = lil.adj
    out[p.nei_sft(p.lil, rc)] = lil.sft
    out[p.nei_spc(p.lil, rc)] = lil.spc
    if set_vec_sod:
        coo_vec_sod = VecSod(
            vec=inp[p.vec(p.coo, rc)],
            sod=inp[p.sod(p.coo, rc)]
        )
        mask = transformation_mask_coo_to_lil(coo)
        out[p.vec(p.lil, rc)] = transform_tensor_coo_to_lil(
            coo_vec_sod.vec, mask, 100)
        out[p.sod(p.lil, rc)] = transform_tensor_coo_to_lil(
            coo_vec_sod.sod, mask, 100)
    return out


class SetAdjSftSpcVecSod(nn.Module):
    def __init__(self, adj, cut: List[Tuple[str, float, bool]]):
        super().__init__()
        self.adj = adj
        self.cut = cut

    def forward(self, inp: Dict[str, Tensor]):
        adj = self.adj(pnt_ful(inp))
        vec = pn.coo2_vec_sod(adj, inp[p.pos], inp[p.cel])
        out = inp.copy()
        for _, rc, set_vs in self.cut:
            out = set_coo_adj_sft_spc_vec_sod(out, adj, vec, rc, set_vs)
        for typ, rc, set_vs in self.cut:
            if typ == p.lil:
                out = set_lil_adj_sft_spc_vec_sod(out, rc, set_vs)
        return out
