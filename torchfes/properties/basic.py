from .cases import add, batch, atoms, saves

idt = 'identity_number'
rst = 'should_reset'
cel = 'cells'
pbc = 'periodic_boundary_condition'
elm = 'elements'
sym = 'symbols'
ent = 'entities'
num = 'number'
num_sqt = 'number_sqrt'
pos = 'positions'
eng = 'total_energies'
eng_mol = 'molecular_energies'
eng_atm = 'atomic_energies'
eng_mol_std = 'molecular_energies_std'
eng_atm_std = 'atomic_energies_std'
eng_mol_ens = 'molecular_energies_ens'
eng_atm_ens = 'atomic_energies_ens'
eng_res = 'restraints_energies'
frc = 'forces_total'
frc_mol = 'forces_from_force_field'
frc_res = 'forces_restraint'
prs = 'pressure_total'
prs_mol = 'pressure_force_field'
prs_res = 'pressure_restraint'
sts = 'stress_total'
sts_mol = 'stress_force_field'
sts_res = 'stress_restraint'

mom = 'momenta'
mas = 'masses'
kbt = 'temperatures'
tim = 'times'
stp = 'steps'
dtm = 'delta_times'
chg = 'charge'

add([batch, saves, atoms], {
    elm, ent, pos, eng_atm, eng_atm_std, eng_atm_ens, frc, frc_mol, frc_res,
    mom, mas, chg
})

add([batch, saves], {
    idt, cel, pbc, num, num_sqt,
    eng, eng_mol, eng_mol_std, eng_mol_ens, eng_res,
    prs, prs_mol, prs_res, sts, sts_mol, sts_res, kbt, tim, stp, dtm
})

add([batch], {rst})
