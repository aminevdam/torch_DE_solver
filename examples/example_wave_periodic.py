import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
# import torch_rbf as rbf
# sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))


from solver import *
from cache import *
from input_preprocessing import *
import time
device = torch.device('cpu')
# Grid
x_grid = np.linspace(0,1,21)
t_grid = np.linspace(0,1,21)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()

grid.to(device)
# Boundary and initial conditions

# u(x,0)=1e4*sin^2(x(x-1)/10)

func_bnd1 = lambda x: 10 ** 4 * np.sin((1/10) * x * (x-1)) ** 2
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bndval1 = func_bnd1(bnd1[:,0])

# du/dx (x,0) = 1e3*sin^2(x(x-1)/10)
func_bnd2 = lambda x: 10 ** 3 * np.sin((1/10) * x * (x-1)) ** 2
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bop2 = {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            }
}
bndval2 = func_bnd2(bnd2[:,0])

# u(0,t) = u(1,t)
bnd3_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)),t).float()
bnd3_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)),t).float()
bnd3 = [bnd3_left,bnd3_right]

# du/dt(0,t) = du/dt(1,t)
bnd4_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)),t).float()
bnd4_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)),t).float()
bnd4 = [bnd4_left,bnd4_right]

bop4= {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            }
}
bcond_type = 'periodic'

bconds = [[bnd1,bndval1],[bnd2,bop2,bndval2],[bnd3,bcond_type],[bnd4,bop4,bcond_type]]

# wave equation is d2u/dt2-(1/4)*d2u/dx2=0
C = 4
wave_eq = {
    'd2u/dt2':
        {
            'coeff': 1,
            'd2u/dt2': [1, 1],
            'pow': 1,
            'var': 0
        },
        '-1/C*d2u/dx2':
        {
            'coeff': -1/C,
            'd2u/dx2': [0, 0],
            'pow': 1,
            'var': 0
        }
}

# NN
model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1))

# unified_bconds = bnd_unify(bconds)
# prepared_grid,grid_dict,point_type = grid_prepare(grid)
# prepared_bconds = bnd_prepare(bconds,prepared_grid,grid_dict,h=0.001)
# full_prepared_operator = operator_prepare(wave_eq, grid_dict, subset=['central'], true_grid=grid, h=0.001)

# plt.scatter(prepared_grid[:,0],prepared_grid[:,1])
# plt.scatter(prepared_grid[prepared_bconds[3][0][0]][:,0],prepared_grid[prepared_bconds[3][0][0]][:,1])
# plt.scatter(prepared_grid[prepared_bconds[0][0]][:,0],prepared_grid[prepared_bconds[0][0]][:,1])
# plt.scatter(prepared_grid[prepared_bconds[3][0][1]][:,0],prepared_grid[prepared_bconds[3][0][1]][:,1])
# plt.show()
start = time.time()

model = point_sort_shift_solver(grid, model, wave_eq , bconds,
                                              lambda_bound=1000, verbose=1, learning_rate=1e-4,
                                    eps=1e-6, tmin=1000, tmax=1e5,use_cache=False,cache_dir='../cache/',cache_verbose=True,
                                    batch_size=None, save_always=True,no_improvement_patience=500,print_every = 500)

end = time.time()
print('Time taken 10= ', end - start)



