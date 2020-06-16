from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

import pointneighbor as pn

from . import properties as p


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


def grad(out: Tensor, inp: Tensor, grd_out: Optional[Tensor],
         create_graph: bool, retain_graph: Optional[bool] = None) -> Tensor:
    grd, = torch.autograd.grad(
        outputs=[out], inputs=[inp], grad_outputs=[grd_out],
        create_graph=create_graph, retain_graph=retain_graph)
    if grd is None:
        raise RuntimeError()
    else:
        return grd


def pnt_ful(inp: Dict[str, Tensor]):
    return pn.pnt_ful(
        cel=inp[p.cel], pbc=inp[p.pbc], pos=inp[p.pos], ent=inp[p.ent])
