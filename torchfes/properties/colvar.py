from .cases import add, batch, saves

col_var = 'colvar'
col_pbc = 'colvar_pbc_length'
col_cen = 'colvar_center'
col_jac = 'colvar_jacobian'
fix_msk = 'fix_mask'
col_mul = 'lagrangian_multiplier'

add([batch, saves], {col_var, col_cen, col_mul, col_jac})
