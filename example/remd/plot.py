import torch
import torchfes as fes
import matplotlib.pyplot as plt

kbt = []
eng = []
with fes.rec.TorchTrajectory('md', 'rb') as rec:
    for mol in rec:
        kbt.append(mol[fes.p.kbt])
        eng_pot = mol[fes.p.eng]
        eng_kin = fes.fn.kinetic_energies(
            mol[fes.p.mom], mol[fes.p.mas], mol[fes.p.ent])
        eng.append(eng_pot + eng_kin)

kbt = torch.stack(kbt)
eng = torch.stack(eng)

for i in range(kbt.size(1)):
    plt.plot(kbt[:, i])
plt.show()
