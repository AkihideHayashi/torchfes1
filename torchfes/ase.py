from ase.units import Bohr
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from .data import from_atoms, to_atoms
from . import properties as p


class TorchFes(Calculator):
    implemented_properties = ['energy', 'forces']
    default_parameters = {'xc': 'ani'}

    nolabel = True

    def __init__(self, evl, order, device):
        Calculator.__init__(self)
        self.evl = evl
        self.order = order
        self.device = device
        self.mol = {}

    def calculate(self, atoms, properties, system_changes):
        self.mol.update(from_atoms(atoms, self.order))
        self.mol = self.evl(self.mol)
        for key in self.mol:
            self.mol[key] = self.mol[key].to(self.device)
        if 'energy' in properties:
            self.results['energy'] = self.mol[p.eng].item()
        if 'forces' in properties:
            self.results['forces'] = self.mol[p.frc].squeeze(0).cpu().numpy()
