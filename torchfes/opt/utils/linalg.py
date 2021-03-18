import torch
from torch import Tensor


def solve(g: Tensor, h: Tensor):
    s, _ = torch.solve(g[:, :, None], h)
    return s.squeeze(2)


def dot(g: Tensor, h: Tensor):
    return (h @ g.unsqueeze(-1)).squeeze(-1)
