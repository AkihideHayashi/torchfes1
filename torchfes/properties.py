import re

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

chg = 'charge'

ads_frq = 'andersen_frequency'

gam_lng = 'langevin_gamma'

omg_nhc = 'nose_hoover_chain_omega'
pos_nhc = 'nose_hoover_chain_positions'
mom_nhc = 'nose_hoover_chain_momenta'
mas_nhc = 'nose_hoover_chain_masses'
con_nhc = 'nose_hoover_chain_constants'

bme_con = 'blue_moon_const'
bme_lmd = 'blue_moon_lambda'
bme_fix = 'blue_moon_fixman_correction_Z^-0.5'
bme_cor = 'blue_moon_correction_kTG'
bme_pup = 'blue_moon_purpose'

mtd_cen = 'metadynamics_potential_center'
mtd_wdt = 'metadynamics_potential_width'
mtd_hgt = 'metadynamics_potential_height'
mtd_gam = 'metadynamics_bias_factor'

coo = 'coo'
lil = 'lil'

_re_float = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
_re_adj_type = r'coo|lil'
is_nei_adj = re.compile(
    r'nei_adj_({})_({})'.format(_re_adj_type, _re_float))
is_nei_sft = re.compile(
    r'nei_sft_({})_({})'.format(_re_adj_type, _re_float))
is_nei_spc = re.compile(
    r'nei_spc_({})_({})'.format(_re_adj_type, _re_float))
is_vec = re.compile(
    r'vec_({})_({})'.format(_re_adj_type, _re_float))
is_sod = re.compile(
    r'sod_({})_({})'.format(_re_adj_type, _re_float))


def nei_adj(typ: str, rc: float):
    return f'nei_adj_{typ}_{rc}'


def nei_sft(typ: str, rc: float):
    return f'nei_sft_{typ}_{rc}'


def nei_spc(typ: str, rc: float):
    return f'nei_spc_{typ}_{rc}'


def vec(typ: str, rc: float):
    return f'vec_{typ}_{rc}'


def sod(typ: str, rc: float):
    return f'sod_{typ}_{rc}'
