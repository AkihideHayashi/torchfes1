from .cases import add, batch, saves


bme_mul = 'blue_moon_lambda'
bme_mul_tmp = 'blue_moon_lambda_tmp'
bme_ktg = 'blue_moon_correction_kTG'
bme_ktg_tmp = 'blue_moon_correction_kTG_tmp'
bme_mmt = 'blue_moon_mass_metrix_tensor'
bme_mmt_det = 'blue_moon_mass_metrix_tensor_determinant'
bme_fix = 'blue_moon_fixman_correction'
bme_fix_tmp = 'blue_moon_fixman_correction_tmp'

add([batch, saves], {
    bme_mul, bme_ktg, bme_mmt, bme_mmt_det, bme_fix
})

add([batch], {bme_mul_tmp, bme_ktg_tmp, bme_fix_tmp})
