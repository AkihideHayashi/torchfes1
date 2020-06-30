import torchfes as fes
from torchfes.recorder import PathPair
from torchfes.data.convert import dict_tensor_to_atoms_list
from torchfes.recorder import not_tmp


with fes.rec.open_torch(PathPair('trj'), 'r') as f:
    data = not_tmp(f[-1])
order = ['H', 'C', 'N', 'O']

atoms_list = dict_tensor_to_atoms_list(data, order)
atoms_list[0].write('test.xyz')
