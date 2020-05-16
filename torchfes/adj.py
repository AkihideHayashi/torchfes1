import logging
from typing import Dict
import torch
from torch import nn, Tensor
import pointneighbor as pn
from pointneighbor import AdjSftSizVecSod
from pointneighbor import (Coo2BookKeeping, Coo2Cel,
                           Coo2FulSimple, Coo2FulPntSft)
from . import properties as p

_logger = logging.getLogger(__name__)


class Adjacent(nn.Module):
    def __init__(self, coo2):
        super().__init__()
        self.coo2 = coo2
        self.pos = torch.tensor([])
        self.cel = torch.tensor([])
        self.adj = torch.tensor([], dtype=torch.int)
        self.sft = torch.tensor([], dtype=torch.int)
        self.siz = torch.tensor([], dtype=torch.int)
        self.vec = torch.tensor([])
        self.sod = torch.tensor([])

    def forward(self, inp: Dict[str, Tensor]):
        if inp[p.pos] is self.pos and inp[p.cel] is self.cel:
            _logger.debug('Adjacent: skip')
            return AdjSftSizVecSod(adj=self.adj, sft=self.sft,
                                   siz=self.siz.tolist(),
                                   vec=self.vec, sod=self.sod)
        else:
            _logger.debug('Adjacent: calc')
            pnt = pn.pnt(cel=inp[p.cel], pbc=inp[p.pbc],
                         pos=inp[p.pos], ent=inp[p.ent])
            pnt_exp = pn.pnt_exp(pnt)
            adj: AdjSftSizVecSod = self.coo2(pnt_exp)
            self.adj = adj.adj
            self.sft = adj.sft
            self.siz = torch.tensor(adj.siz)
            self.vec = adj.vec
            self.sod = adj.sod
            self.pos = inp[p.pos]
            self.cel = inp[p.cel]
            return adj


def book_keeping(model: nn.Module, rc: float, delta: float) -> Adjacent:
    return Adjacent(Coo2BookKeeping(model, rc, delta))


__all__ = [
    "AdjSftSizVecSod",
    "Coo2BookKeeping",
    "Coo2Cel",
    "Coo2FulPntSft",
    "Coo2FulSimple",
    "Adjacent",
    "book_keeping",
]
