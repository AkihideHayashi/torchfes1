from .cases import add, batch, saves


ads_frq = 'andersen_frequency'
gam_lng = 'langevin_gamma'

omg_nhc = 'nose_hoover_chain_omega'
pos_nhc = 'nose_hoover_chain_positions'
mom_nhc = 'nose_hoover_chain_momenta'
mas_nhc = 'nose_hoover_chain_masses'
con_nhc = 'nose_hoover_chain_constants'

add([batch, saves], {
    ads_frq, gam_lng, omg_nhc, pos_nhc, mom_nhc, mas_nhc, con_nhc
})
