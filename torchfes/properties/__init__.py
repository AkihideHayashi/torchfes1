# flake8: noqa
from math import nan
from .basic import *
from .opt import *
from .bme import *
from .mtd import *
from .adj import *
from .thermostat import *
from .cases import atoms, batch, saves, metad
from .constraints import *


default_values = {
    pos: 0.0,
    elm: -1,
    mas: 1.0,
    frc: 0.0,
    frc_mol: 0.0,
    frc_res: 0.0,
    eng_atm: 0.0,
    eng_atm_std: 0.0,
    eng_atm_ens: 0.0,
    ent: False,
    mom: 0.0,
    chg: 0.0,
    mtd_cen: 0.0,
    mtd_hgt: 0.0,
    mtd_prc: 0.0,
    con_frc: 0.0,
    con_jac: 0.0,
    fix_msk: True,
    eig_vec: 0.0,
    dim: 0.0,
}

for key in atoms:
    assert key in default_values, key
