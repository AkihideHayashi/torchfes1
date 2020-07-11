from typing import Dict, List
from torch import Tensor
from .. import properties as p

default_mask_keys = [p.idt, p.sld_rst, p.cel, p.pbc, p.elm, p.sym,
                     p.ent, p.pos,
                     p.eng, p.eng_mol, p.eng_atm,
                     p.eng_mol_std, p.eng_atm_std, p.eng_res,
                     p.frc, p.frc_mol, p.frc_res,
                     p.prs, p.prs_mol, p.prs_res,
                     p.sts, p.sts_mol, p.sts_res,
                     p.mom, p.mas, p.kbt, p.tim, p.stp, p.dtm, p.chg,
                     p.ads_frq,
                     p.gam_lng,
                     p.omg_nhc, p.pos_nhc, p.mom_nhc,
                     p.mas_nhc, p.con_nhc, p.fir_cnt,
                     p.fir_alp,
                     p.bme_cen, p.bme_lmd, p.bme_lmd_tmp, p.bme_jac_con_pos,
                     p.bme_ktg, p.bme_ktg_tmp, p.bme_mmt, p.bme_mmt_det,
                     p.bme_fix, p.bme_fix_tmp,
                     p.res_cen,
                     p.mtd_dep_cen, p.mtd_dep_gam, p.mtd_dep_hgt, p.mtd_dep_prc
                     ]


def masked_select(mol: Dict[str, Tensor], mask: Tensor, keys: List[str]):
    ret: Dict[str, Tensor] = {}
    for key in mol:
        if key not in keys:
            ret[key] = mol[key]
        else:
            ret[key] = mol[key][mask]
    return ret


def masked_scatter(mol: Dict[str, Tensor], mask: Tensor,
                   source: Dict[str, Tensor], keys: List[str]):
    ret: Dict[str, Tensor] = {}
    for key in source.keys():
        if key not in keys:
            ret[key] = source[key]
        else:
            tmp = mask.clone()
            for i in range(mol[key].dim() - 1):
                tmp.unsqueeze_(-1)
            ret[key] = mol[key].masked_scatter(tmp, source[key])
    return ret
