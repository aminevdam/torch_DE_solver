import torch
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions


domain = Domain()

domain.variable('x', [0,1], 3)
domain.variable('t', [1,2], 5)

grid = domain.build('NN')

# print(grid)
boundaries = Conditions()

bop= {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            }
}

boundaries.operator({'x':torch.tensor([0]), 't': torch.tensor([1, 1.1, 1.2, 2])}, operator=bop, value=5)
boundaries.periodic([{'x':0, 't':[0,2]}, {'x':1, 't':[1,2]}], bop)

bconds = boundaries.build(domain.variable_dict)

print(bconds)