from .cases import add, batch, saves

fir_cnt = 'fire_count'
fir_alp = 'fire_a'
gen_eng = 'general_energy'  # should be Langrangian
gen_pos = 'general_position'
gen_grd = 'general_gradient'
gen_stp = 'general_step'  # line search
gen_dir = 'general_direction'
gen_dir_grd = 'genaral_directional_differential'  # line search
gen_dir_hes = 'genaral_directional_hessian'  # line search
gen_vec = 'general_vector'  # gen_dir * gen_stp
gen_hes = 'general_hessian'
gen_hes_inv = 'general_hessian_inverse'
gen_dlt_pos = 'general_delta_position'  # LBFGS
gen_dlt_grd = 'general_delta_gradient'  # LBFGS
gen_dlt_dot = 'general_delta_dot'  # LBFGS
fix_msk = 'fix_mask'

add([saves, batch], {fir_cnt, fir_alp})
