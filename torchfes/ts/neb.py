from math import pi
from torchfes.ts.utils import spring_energy
from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p


# class NudgedElasticBand(nn.Module):

#     def __init__(self, k: float, nudged=1.0):
#         super().__init__()
#         self.k = torch.tensor(k)
#         self.nudged = nudged

#     def forward(self, inp: Dict[str, Tensor], create_graph, retain_graph):
#         pos = inp[p.pos]
#         frc = inp[p.frc]
#         pos_mns = pos[:-2, :, :]
#         pos_pls = pos[2:, :, :]
#         pos_zer = pos[1:-1, :, :]
#         vec_pls = pos_pls - pos_zer
#         vec_mns = pos_zer - pos_mns

#         if self.nudged == 1.0:
#             frc_neb = nudged_elastic_band(
#                 frc[1:-1, :, :], vec_pls, vec_mns, self.k)
#         elif self.nudged == 0.0:
#             frc_neb = elastic_band(
#                 frc[1:-1, :, :], vec_pls, vec_mns, self.k)
#         else:
#             frc_neb = (
#                 self.nudged * nudged_elastic_band(
#                     frc[1:-1, :, :], vec_pls, vec_mns, self.k) +
#                 (1 - self.nudged) * elastic_band(
#                     frc[1:-1, :, :], vec_pls, vec_mns, self.k))
#         frc = frc + torch.nn.functional.pad(frc_neb, [0, 0, 0, 0, 1, 1])
#         out = inp.copy()
#         out[p.frc] = frc
#         return out


# def unit(vec: Tensor):
#     return vec / dot(vec, vec).sqrt()


# def dot(a: Tensor, b: Tensor):
#     return (a * b).sum(dim=-2, keepdim=True).sum(dim=-1, keepdim=True)


# def norm(a: Tensor):
#     return dot(a, a).sqrt()


# def nudged_elastic_band(frc, vecp, vecm, k):
#     tau = unit(unit(vecp) + unit(vecm))
#     term1 = dot(frc, tau) * tau
#     term2 = k * dot(vecp - vecm, tau) * tau
#     return - term1 + term2


# def elastic_band(frc, vecp, vecm, k):
#     return (vecp - vecm) * k


# def switching_doubly_nudged_elastic_band(frc, vecp, vecm, k):
#     tau = unit(unit(vecp) + unit(vecm))
#     spr = k * (vecp - vecm)
#     frc_para = dot(frc, tau) * tau
#     spr_para = k * (norm(vecp) - norm(vecm)) * tau
#     frc_perp = frc - frc_para
#     spr_perp = spr - spr_para
#     doubly = spr_perp - dot(spr_perp, frc_perp) * frc_perp
#     sw = 2 / pi * torch.atan2(dot(frc_perp, frc_perp), dot(spr_perp, spr_perp))
#     assert (sw > 0).all()
#     return -frc_para + spr_para + doubly * sw


# def doubly_nudged_elastic_band(frc, vecp, vecm, k):
#     tau = unit(unit(vecp) + unit(vecm))
#     spr = k * (vecp - vecm)
#     frc_para = dot(frc, tau) * tau
#     spr_para = k * (norm(vecp) - norm(vecm)) * tau
#     frc_perp = frc - frc_para
#     spr_perp = spr - dot(spr, tau) * tau
#     doubly = spr_perp - dot(spr_perp, frc_perp) * frc_perp
#     return -frc_para + spr_para + doubly


# class DoublyNudgedElasticBand(nn.Module):

#     def __init__(self, k: float, sw: bool):
#         super().__init__()
#         self.k = torch.tensor(k)
#         self.sw = sw

#     def forward(self, inp: Dict[str, Tensor], create_graph, retain_graph):
#         pos = inp[p.pos]
#         frc = inp[p.frc]
#         pos_mns = pos[:-2, :, :]
#         pos_pls = pos[2:, :, :]
#         pos_zer = pos[1:-1, :, :]
#         vec_pls = pos_pls - pos_zer
#         vec_mns = pos_zer - pos_mns
#         if self.sw:
#             frc_neb = switching_doubly_nudged_elastic_band(
#                 frc[1:-1, :, :], vec_pls, vec_mns, self.k)
#         else:
#             frc_neb = doubly_nudged_elastic_band(
#                 frc[1:-1, :, :], vec_pls, vec_mns, self.k)
#         frc = frc + torch.nn.functional.pad(frc_neb, [0, 0, 0, 0, 1, 1])
#         out = inp.copy()
#         out[p.frc] = frc
#         return out


# def doubly_climbing_nudged_elastic_band(eng, frc, vecp, vecm, k):
#     tau = unit(unit(vecp) + unit(vecm))
#     spr = k * (vecp - vecm)
#     frc_para = dot(frc, tau) * tau
#     spr_para = k * (norm(vecp) - norm(vecm)) * tau
#     print(norm(vecp).size())
#     print(torch.max(eng[1:], eng[:-1]).size())
#     frc_perp = frc - frc_para
#     spr_perp = spr - dot(spr, tau) * tau
#     doubly = spr_perp - dot(spr_perp, frc_perp) * frc_perp
#     i = eng.argmax()
#     frc_para[i, :, :] *= 2
#     return -frc_para + spr_para + doubly


# class DoublyClimbingNudgedElasticBand(nn.Module):

#     def __init__(self, k: float):
#         super().__init__()
#         self.k = torch.tensor(k)

#     def forward(self, inp: Dict[str, Tensor], create_graph, retain_graph):
#         pos = inp[p.pos]
#         frc = inp[p.frc]
#         pos_mns = pos[:-2, :, :]
#         pos_pls = pos[2:, :, :]
#         pos_zer = pos[1:-1, :, :]
#         vec_pls = pos_pls - pos_zer
#         vec_mns = pos_zer - pos_mns
#         eng = inp[p.eng][1:-1]
#         frc_neb = doubly_climbing_nudged_elastic_band(
#             eng, frc[1:-1, :, :], vec_pls, vec_mns, self.k)
#         frc = frc + torch.nn.functional.pad(frc_neb, [0, 0, 0, 0, 1, 1])
#         out = inp.copy()
#         out[p.frc] = frc
#         return out


# def nudged_elastic_band(eng, frc, vecp, vecm, k):
#     tau = unit(unit(vecp) + unit(vecm))
#     spr = k * (vecp - vecm)
#     frc_para = dot(frc, tau) * tau
#     spr_para = k * (norm(vecp) - norm(vecm)) * tau
#     frc_perp = frc - frc_para
#     spr_perp = spr - dot(spr, tau) * tau
#     doubly = spr_perp - dot(spr_perp, frc_perp) * frc_perp
#     i = eng.argmax()
#     frc_para[i, :, :] *= 2
#     return -frc_para + spr_para + doubly

def unit(vec: Tensor):
    return vec / dot(vec, vec).sqrt()


def dot(a: Tensor, b: Tensor):
    return (a * b).sum(dim=-2, keepdim=True).sum(dim=-1, keepdim=True)


def norm(a: Tensor):
    return dot(a, a).sqrt()


def normalize(a: Tensor):
    return a / norm(a)


def improved_tangent_estimation(eng: Tensor, pos: Tensor):
    tau_p = pos[1:-1] - pos[:-2]
    tau_m = pos[2:] - pos[1:-1]
    eng_msk_pp = (eng[2:] > eng[1:-1]) & (eng[1:-1] > eng[:-2])
    eng_msk_mm = (eng[2:] < eng[1:-1]) & (eng[1:-1] < eng[:-2])
    eng_msk_p = (eng[2:] > eng[1:-1]) & ~(eng_msk_pp | eng_msk_mm)
    eng_msk_m = (eng[2:] < eng[1:-1]) & ~(eng_msk_pp | eng_msk_mm)
    dv_max = torch.max(
        torch.abs(eng[2:] - eng[1:-1]), torch.abs(eng[:-2] - eng[1:-1]))
    dv_min = torch.min(
        torch.abs(eng[2:] - eng[1:-1]), torch.abs(eng[:-2] - eng[1:-1]))
    tau = (tau_p * eng_msk_pp
           + tau_m * eng_msk_mm
           + (tau_p * dv_max + tau_m * dv_min) * eng_msk_p
           + (tau_p * dv_min + tau_m * dv_max) * eng_msk_m)
    return tau


def spring_force(eng: Tensor, k_max: float, dlt_k: float, eps: float):
    eng_ref = eng.min() - eps
    eng_max = eng.max()
    eng_i = torch.max(eng[1:], eng[:-1])
    k = k_max - dlt_k * (eng_max - eng_i) / (eng_max - eng_ref)
    k.masked_fill_(eng_i < eng_ref, k_max - dlt_k)
    return k


class ElasticBand(nn.Module):

    def __init__(self, k: float, dk: float,
                 nudged=True, climbing=True, doubly=True):
        super().__init__()
        self.k = k
        self.dk = dk
        self.eps = 0.1
        self.nudged = nudged
        self.climbing = climbing
        self.doubly = doubly

    def forward(self, inp: Dict[str, Tensor], create_graph, retain_graph):
        pos = inp[p.pos]
        frc = inp[p.frc][1:-1]
        eng = inp[p.eng][:, None, None]
        k = spring_force(eng, self.k, self.dk, self.eps)

        tau = normalize(improved_tangent_estimation(eng, pos))
        vecp = pos[2:] - pos[1:-1]
        vecm = pos[1:-1] - pos[:-2]

        frc_para = dot(frc, tau) * tau
        spr = k[1:] * vecp - k[:-1] * vecm
        spr_para = (k[1:] * norm(vecp) - k[:-1] * norm(vecm)) * tau
        frc_perp = frc - frc_para
        spr_perp = spr - spr_para
        if self.climbing:
            i = eng.flatten().argmax()
            frc_para[i, :, :] *= 2
        if self.nudged:
            frc_neb = spr_para - frc_para
        else:
            frc_neb = spr
        if self.doubly:
            sw = 2 / pi * torch.atan2(dot(frc_perp, frc_perp), dot(spr_perp, spr_perp))
            doubly = spr_perp - dot(spr_perp, frc_perp) * frc_perp * sw
            frc_neb = frc_neb + doubly

        out = inp.copy()
        out[p.frc] = out[p.frc] + \
            torch.nn.functional.pad(frc_neb, [0, 0, 0, 0, 1, 1])
        return out
