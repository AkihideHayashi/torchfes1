from .cases import add, batch, saves, atoms

fir_cnt = 'fire_count'
fir_alp = 'fire_a'
eng_prd = 'predicted_energy'
gen_pos = 'general_position'
gen_grd = 'general_gradient'
gen_stp_siz = 'general_step_size'  # line search
gen_stp_dir = 'general_direction'
gen_stp = 'general_step'  # gen_dir * gen_siz
gen_dir_grd = 'general_directional_gradients'
gen_dir_hes = 'general_directional_hessian'
gen_lin_tol = 'general_linesearch_tolerance'
gen_hes = 'general_hessian'
gen_hes_inv = 'general_hessian_inverse'
gen_pos_pre = 'previous_general_position'
gen_grd_pre = 'previous_general_gradient'
gen_dlt_pos = 'general_delta_position'
gen_dlt_grd = 'general_delta_gradient'
gen_dlt_dot = 'general_delta_dot'
con_aug = 'constraint_augumented_langrangian'
dim = 'dimer_vector'

add([saves, batch], {fir_cnt, fir_alp, con_aug})
add([saves, batch, atoms], {dim})
