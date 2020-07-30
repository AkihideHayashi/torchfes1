from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn

import pointneighbor as pn

from . import properties as p


class Sequential(nn.ModuleList):
    def __init__(self, *args):
        super().__init__(args)

    def forward(self, mol: Dict[str, Tensor]):
        for mod in self:
            mol = mod(mol)
        return mol


def sym_to_elm(symbols: Union[str, List, np.ndarray],
               order: Union[np.ndarray, List[str]]):
    """Transform symbols to elements."""
    if not isinstance(order, list):
        order = order.tolist()
    if not isinstance(symbols, (str, list)):
        symbols = symbols.tolist()
    if isinstance(symbols, str):
        if symbols in order:
            return order.index(symbols)
        else:
            return -1
    else:
        return np.array([sym_to_elm(s, order) for s in symbols])


def detach_(inp: Dict[str, Tensor]):
    for key in inp:
        inp[key] = inp[key].clone().detach()


def detach(inp: Dict[str, Tensor]):
    out = inp.copy()
    detach_(out)
    return out


def grad(out: Tensor, inp: Tensor, grd_out: Optional[Tensor] = None,
         create_graph: bool = False, retain_graph: Optional[bool] = None
         ) -> Tensor:
    if grd_out is None:
        grd_out = torch.ones_like(out)
    return _grad_inner(out, inp, grd_out, create_graph, retain_graph)


def _grad_inner(out: Tensor, inp: Tensor, grd_out: Optional[Tensor],
                create_graph: bool, retain_graph: Optional[bool]) -> Tensor:
    if not out.requires_grad:
        return torch.zeros_like(inp)
    grd, = torch.autograd.grad(
        outputs=[out], inputs=[inp], grad_outputs=[grd_out],
        create_graph=create_graph, retain_graph=retain_graph,
        allow_unused=True)
    if grd is None:
        return torch.zeros_like(inp)
    else:
        return grd


def requires_grad(inp: Dict[str, Tensor], props: List[str]):
    out = inp.copy()
    for prop in props:
        out[prop] = inp[prop].clone().detach().requires_grad_(True)
    return out


def pnt_ful(inp: Dict[str, Tensor]):
    return pn.pnt_ful(
        cel_mat=inp[p.cel], pbc=inp[p.pbc], pos_xyz=inp[p.pos], ent=inp[p.ent])
