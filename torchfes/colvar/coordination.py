from math import inf
from typing import Dict, Tuple, List
import torch
from torch import nn, Tensor
import pointneighbor as pn
from ..adj import get_adj_sft_spc, vec_sod
from .. import properties as p


def ravel1(idx: List[Tensor], siz: List[int]):
    return pn.fn.ravel1(
        torch.stack(idx), torch.tensor(siz, device=idx[0].device), dim=0)


def smap(a, c, d, rd):
    return (1.0 + c * rd ** a) ** d


def smap_c(a, b):
    return 2 ** (a / b) - 1


def smap_d(a, b):
    return - b / a


class Smap(nn.Module):
    a: Tensor
    b: Tensor
    c: Tensor
    d: Tensor
    r0: Tensor
    d0: Tensor

    def __init__(self, d0, r0, a, b):
        super().__init__()
        self.register_buffer('d0', d0)
        self.register_buffer('r0', r0)
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        self.register_buffer('c', smap_c(self.a, self.b))
        self.register_buffer('d', smap_d(self.a, self.b))

    def forward(self, eij, dst):
        rd = (dst - self.d0[eij]) / self.r0[eij]
        ret = smap(self.a[eij], self.c[eij], self.d[eij], rd
                   ).masked_fill(eij < 0, 0.0)
        return ret.masked_fill(rd < 0, 1.0)


def rational_almost(rd, nn, nd):
    num = 1 - rd.pow(nn)
    den = 1 - rd.pow(nd)
    return num / den


def rational_singularity(rd, nn, nd):
    return 0.5 * nn * (2 + (nn - nd) * (rd - 1)) / nd


def _no_nan(x: Tensor):
    return (x == x).all()


class Rational(nn.Module):
    def __init__(self, d0, r0, nn, nd):
        super().__init__()
        self.register_buffer('d0', d0)
        self.register_buffer('r0', r0)
        self.register_buffer('nn', nn)
        self.register_buffer('nd', nd)
        self.eps = 1e-2

    def forward(self, eij, dst):
        assert _no_nan(dst)
        rd = (dst - self.d0[eij]) / self.r0[eij]
        nn = self.nn[eij]
        nd = self.nd[eij]
        rat_almost = rational_almost(rd, nn, nd)
        rat_singul = rational_singularity(rd, nn, nd)
        sing = (rd < 1 + self.eps) & (rd > 1 - self.eps)
        rat = torch.where(sing, rat_singul, rat_almost)
        ret = rat.masked_fill(rd < 0, 1.0)
        assert _no_nan(ret)
        ret.masked_fill_(eij < 0, 0.0)
        return ret


def mollifier_inner(x, a, rc):
    return (
        1
        - torch.exp(- a / (rc * rc - (rc - x) ** 2)) / torch.exp(-a / rc / rc)
    )


def mollifier_outer(x: Tensor, rc: Tensor) -> Tensor:
    return (x <= 0).to(x)


def mollifier(x: Tensor, a: Tensor, rc: Tensor):
    mask = (x > 1e-6) & (x < rc - 1e-6)
    outer = mollifier_outer(x, rc)
    inner = mollifier_inner(x[mask], a[mask], rc[mask])
    return outer.masked_scatter(mask, inner)


class Mollifier(nn.Module):
    a: Tensor
    d0: Tensor
    rc: Tensor

    def __init__(self, a, d0, rc):
        super().__init__()
        self.register_buffer('a', a)
        self.register_buffer('d0', d0)
        self.register_buffer('rc', rc)

    def forward(self, eij, dst):
        rc = self.rc[eij]
        d0 = self.d0[eij]
        a = self.a[eij]
        return mollifier(dst - d0, a, rc - d0)


class Coordination(nn.Module):
    elm: Tensor
    coef: Tensor

    def __init__(self, mod, numel: int, rc: float,
                 items: Dict[Tuple[int, int], Dict[str, float]]):
        super().__init__()
        elm = -torch.ones([numel, numel], dtype=torch.long)
        dic: Dict[str, List[float]] = {}
        n = 0
        for n, ((i, j), prp) in enumerate(items.items()):
            elm[i, j] = n
            elm[j, i] = n
            for key, val in prp.items():
                if key not in dic:
                    dic[key] = []
                dic[key].append(val)
        self.register_buffer('elm', elm)
        self.mod = mod(**{key: torch.tensor(val) for key, val in dic.items()})
        self.rc = rc
        self.n = n + 1
        self.pbc = torch.full([self.n], inf)

    def forward(self, inp: Dict[str, Tensor]):
        adj = get_adj_sft_spc(inp, p.coo, self.rc)
        n, i, j = pn.coo2_n_i_j(adj)
        ei = inp[p.elm][n, i]
        ej = inp[p.elm][n, j]
        eij = self.elm[ei, ej]
        adapt = eij >= 0
        _, sod = vec_sod(inp, adj)
        dis: Tensor = sod[adapt].sqrt()
        eij = eij[adapt]
        coords = self.mod(eij, dis)
        n_bch, _ = inp[p.elm].size()
        idx = ravel1([n[adapt], eij], [n_bch, self.n])
        coord = torch.zeros([n_bch * self.n],
                            dtype=sod.dtype, device=sod.device)
        coord.index_add_(0, idx, coords)
        ret = coord.view([n_bch, self.n]) / 2
        return ret


class SlabCoordination(nn.Module):
    elm: Tensor
    coef: Tensor
    wz: Tensor
    wr: Tensor
    pbc: Tensor

    def __init__(self, mod, numel: int, rc: float,
                 items: Dict[
                     Tuple[int, int],
                     Tuple[List[float], Dict[str, float]]], dim: int = 2):
        super().__init__()
        elm = -torch.ones([numel, numel], dtype=torch.long)
        dic: Dict[str, List[float]] = {}
        wz = []
        wr = []
        n = 0
        for n, ((i, j), (coef, prp)) in enumerate(items.items()):
            elm[i, j] = n
            for key, val in prp.items():
                if key not in dic:
                    dic[key] = []
                dic[key].append(val)
            wz.append(coef[0])
            wr.append(coef[1])
        self.register_buffer('elm', elm)
        self.mod = mod(**{key: torch.tensor(val) for key, val in dic.items()})
        self.rc = rc
        self.n = n + 1
        self.register_buffer('pbc', torch.full([self.n], inf))
        self.register_buffer('wz', 0.5 / torch.tensor(wz).pow(2))
        self.register_buffer('wr', 0.5 / torch.tensor(wr).pow(2))
        self.dim = dim

    def forward(self, inp: Dict[str, Tensor]):
        num_bch = inp[p.elm].size(0)
        adj = get_adj_sft_spc(inp, p.coo, self.rc)
        n, i, j = pn.coo2_n_i_j(adj)
        ei = inp[p.elm][n, i]
        ej = inp[p.elm][n, j]
        eij = self.elm[ei, ej]
        adapt = eij >= 0
        vec, sod = vec_sod(inp, adj)
        n, i, j = n[adapt], i[adapt], j[adapt]
        vec, sod = vec[adapt], sod[adapt]
        ei, ej, eij = ei[adapt], ej[adapt], eij[adapt]
        zij = -vec[:, self.dim]
        wij = torch.exp(-self.wz[eij] * zij) * torch.exp(-self.wr[eij] * sod)
        i_max = i.max() + 5
        ni = n * i_max + i
        unique, idx, cou = torch.unique_consecutive(
            ni, return_inverse=True, return_counts=True)
        cum = pn.fn.cumsum_from_zero(cou)
        den = torch.zeros_like(unique, dtype=wij.dtype)
        den.index_add_(0, idx, wij)
        num = torch.zeros_like(unique, dtype=wij.dtype)
        num.index_add_(0, idx, wij * zij)
        zij_ = num / den
        eij_ = eij[cum]
        n_ = n[cum]
        cij_ = self.mod(eij_, zij_)
        idx_ = n_ * self.n + eij_
        ret = torch.zeros([num_bch * self.n],
                          device=n.device, dtype=cij_.dtype)
        ret.index_add_(0, idx_, cij_)
        return ret.view([num_bch, self.n])
