from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn


class PosEngFrc(NamedTuple):
    pos: Tensor
    eng: Tensor
    frc: Tensor


class PosEngFrcStorage(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = torch.tensor([])
        self.eng = torch.tensor([])
        self.frc = torch.tensor([])

    def forward(self, inp: Optional[PosEngFrc] = None):
        if inp is not None:
            self.pos = inp.pos
            self.eng = inp.eng
            self.frc = inp.frc
        return PosEngFrc(pos=self.pos, eng=self.eng, frc=self.frc)


def where_pef(flt: Tensor, ift: PosEngFrc, iff: PosEngFrc):
    return PosEngFrc(
        pos=torch.where(flt, ift.pos, iff.pos),
        eng=torch.where(flt, ift.eng, iff.eng),
        frc=torch.where(flt, ift.frc, iff.frc),
    )
