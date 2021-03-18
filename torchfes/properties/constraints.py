from .cases import batch, saves, add, atoms
# con_frc[bch, atm, dim] =
#    -sum(con_jac[bch, atm, dim, con] * con_mul[bch, con], con)
# con_frc[bch, atm, dim]

con_mul = 'constraint_lagrangian_multiplier'
con_jac = 'constraint_jacobian'
con_frc = 'constraint_forces'
con_cen = 'constraint_center'

add([batch, saves], {con_mul, con_jac, con_cen})
add([batch, saves, atoms], {con_frc})
