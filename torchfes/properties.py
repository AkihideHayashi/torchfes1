"""You should renew recorder.selector after edit this file."""
import re

idt = 'identity_number'
tmp = 'tmp'
sld_rst = f'should_reset_{tmp}'
cel = 'cells'
pbc = 'periodic_boundary_condition'
elm = 'elements'
sym = 'symbols'
ent = 'entities'
pos = 'positions'
eng = 'total_energies'
eng_mol = 'molecular_energies'
eng_atm = 'atomic_energies'
eng_mol_std = 'molecular_energies_std'
eng_atm_std = 'atomic_energies_std'
eng_mol_ens = 'molecular_energies_ens'
eng_atm_ens = 'atomic_energies_ens'
eng_res = 'restraints_energies'
frc = 'forces_total'
frc_mol = 'forces_from_force_field'
frc_res = 'forces_restraint'
prs = 'pressure_total'
prs_mol = 'pressure_force_field'
prs_res = 'pressure_restraint'
sts = 'stress_total'
sts_mol = 'stress_force_field'
sts_res = 'stress_restraint'
mom = 'momenta'
mas = 'masses'
kbt = 'temperatures'
tim = 'times'
stp = 'steps'
dtm = 'delta_times'
fix = 'fix'

chg = 'charge'

ads_frq = 'andersen_frequency'

gam_lng = 'langevin_gamma'

omg_nhc = 'nose_hoover_chain_omega'
pos_nhc = 'nose_hoover_chain_positions'
mom_nhc = 'nose_hoover_chain_momenta'
mas_nhc = 'nose_hoover_chain_masses'
con_nhc = 'nose_hoover_chain_constants'

fir_cnt = 'fire_count'
fir_alp = 'fire_a'

bme_cen = 'blue_moon_center'
bme_lmd = 'blue_moon_lambda'
bme_lmd_tmp = f'blue_moon_lambda_{tmp}'
bme_jac_con_pos = 'blue_moon_jacobian_of_constraint'
bme_ktg = 'blue_moon_correction_kTG'
bme_ktg_tmp = f'blue_moon_correction_kTG_{tmp}'
bme_mmt = 'blue_moon_mass_metrix_tensor'
bme_mmt_det = 'blue_moon_mass_metrix_tensor_determinant'
bme_fix = 'blue_moon_fixman_correction'
bme_fix_tmp = f'blue_moon_fixman_correction_{tmp}'

res_cen = 'restrained_purpose_sigma_0'

mtd_cen = f'metadynamics_potential_center_{tmp}'
mtd_prc = f'metadynamics_precision_matrix_{tmp}'
mtd_hgt = f'metadynamics_potential_height_{tmp}'
mtd_gam = f'metadynamics_bias_factor_{tmp}'

mtd_dep_cen = 'metadynamics_potential_center'
mtd_dep_prc = 'metadynamics_precision_matrix'
mtd_dep_hgt = 'metadynamics_potential_height'
mtd_dep_gam = 'metadynamics_bias_factor'

coo = 'coo'
lil = 'lil'

_re_float = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
_re_adj_type = r'coo|lil'
is_nei_adj = re.compile(
    r'nei_adj_({})_({})_{}'.format(_re_adj_type, _re_float, tmp))
is_nei_sft = re.compile(
    r'nei_sft_({})_({})_{}'.format(_re_adj_type, _re_float, tmp))
is_nei_spc = re.compile(
    r'nei_spc_({})_({})_{}'.format(_re_adj_type, _re_float, tmp))
is_vec = re.compile(
    r'vec_({})_({})_{}'.format(_re_adj_type, _re_float, tmp))
is_sod = re.compile(
    r'sod_({})_({})_{}'.format(_re_adj_type, _re_float, tmp))


def nei_adj(typ: str, rc: float):
    return f'nei_adj_{typ}_{rc}_tmp'


def nei_sft(typ: str, rc: float):
    return f'nei_sft_{typ}_{rc}_tmp'


def nei_spc(typ: str, rc: float):
    return f'nei_spc_{typ}_{rc}_tmp'


def vec(typ: str, rc: float):
    return f'vec_{typ}_{rc}_tmp'


def sod(typ: str, rc: float):
    return f'sod_{typ}_{rc}_tmp'


def is_tmp(key: str):
    return key.split('_')[-1] == tmp


def is_mtd(key: str):
    return key.split('_')[0] == 'metadynamics'
