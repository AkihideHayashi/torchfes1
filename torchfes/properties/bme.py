from .cases import add, batch, save_trj, atoms


bme_lmd = 'blue_moon_lambda'
bme_lmd_tmp = 'blue_moon_lambda_tmp'
bme_jac_con_pos = 'blue_moon_jacobian_of_constraint'
bme_ktg = 'blue_moon_correction_kTG'
bme_ktg_tmp = 'blue_moon_correction_kTG_tmp'
bme_mmt = 'blue_moon_mass_metrix_tensor'
bme_mmt_det = 'blue_moon_mass_metrix_tensor_determinant'
bme_fix = 'blue_moon_fixman_correction'
bme_fix_tmp = 'blue_moon_fixman_correction_tmp'
bme_frc = 'blue_moon_forces'

res_cen = 'restrained_purpose_sigma_0'

add([batch, save_trj, atoms], {bme_frc})
add([batch, save_trj], {
    bme_lmd, bme_jac_con_pos, bme_ktg, bme_mmt, bme_mmt_det, bme_fix, res_cen
})

add([batch], {bme_lmd_tmp, bme_ktg_tmp, bme_fix_tmp})
