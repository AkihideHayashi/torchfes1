from .cases import add, batch, saves

fir_cnt = 'fire_count'
fir_alp = 'fire_a'
gen_pos = 'general_position'
gen_grd = 'general_gradient'
gen_stp = 'general_step'  # line search
gen_dir = 'general_direction'
gen_vec = 'general_vector'  # gen_dir * gen_stp
gen_hes = 'general_hessian'
gen_hes_inv = 'general_hessian_inverse'
gen_dlt_pos = 'general_delta_position'
gen_dlt_grd = 'general_delta_gradient'
gen_dlt_dot = 'general_delta_dot'
con_aug = 'constraint_augumented_langrangian'
con_mul = 'constraint_langrangian_multiplier'
con_mul_frc = 'constraint_langrangian_multiplier_force'
con_cen = 'constraint_center'
hes = 'hessian'
hes_inv = 'hessian_inverse'

add([saves, batch], {fir_cnt, fir_alp, con_mul, con_cen, con_aug})
add([batch], {hes, hes_inv})
