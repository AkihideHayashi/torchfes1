"""Wolfe Condition."""
import torch
from torch import nn, Tensor
from ..general import PosEngFrc


def _dot(a, b):
    return (a * b).sum(-1).unsqueeze(-1)


class WolfeCondition(nn.Module):
    """
    Args:
        old: old pos, eng, and frc.
        new: line search new point.
        old.pos + stp * vec == new.pos
    Returns:
        ret > 0 -> stp is too big.
        ret < 0 -> stp is too small.
        ret = 0 -> stp satisfies wolfe's condition.
    """

    def __init__(self, c1: float, c2: float, eps1: float, eps2: float):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.eps1 = eps1
        self.eps2 = eps2
        self.con = torch.tensor([], dtype=torch.long)
        assert 0 < c1 < c2 < 1
    
    def peek(self):
        return self.con

    def forward(self, old: PosEngFrc, new: PosEngFrc,
                stp: Tensor, vec: Tensor):
        assert torch.allclose(old.pos + stp * vec, new.pos), (
            (vec == vec).all(), (new.pos == new.pos).all())
        # big = new.eng > old.eng - self.c1 * stp * _dot(old.frc, vec)
        e1 = self.eps1
        e2 = self.eps2
        big = new.eng > old.eng - self.c1 * stp * _dot(old.frc, vec) + e1
        sml = _dot(new.frc, vec) > self.c2 * _dot(old.frc, vec) + e2
        bad = big & sml
        self.con = (
            big.to(torch.long)
            - sml.to(torch.long)
            + bad.to(torch.long)
        ).to(new.eng)
        return self.con
