from .cases import add, batch, save_trj, metadynamics

mtd_ens_cen = 'metadynamics_ensemble_potential_center'
mtd_ens_prc = 'metadynamics_ensemble_precision_matrix'
mtd_ens_hgt = 'metadynamics_ensemble_potential_height'
mtd_ens_gam = 'metadynamics_ensemble_bias_factor'

mtd_dep_cen = 'metadynamics_potential_center'
mtd_dep_prc = 'metadynamics_precision_matrix'
mtd_dep_hgt = 'metadynamics_potential_height'
mtd_dep_gam = 'metadynamics_bias_factor'

add([metadynamics], {mtd_ens_cen, mtd_ens_prc, mtd_ens_hgt, mtd_ens_gam})
add([batch, save_trj], {mtd_dep_cen, mtd_dep_prc, mtd_dep_hgt, mtd_dep_gam})
