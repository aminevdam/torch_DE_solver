import torch
import numpy as np
import pandas as pd
import sys
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import EarlyStopping, Plots, Cache
from tedeous.optimizers.optimizer import Optimizer

devices = ['cpu', 'cuda']
mixed_precision = [True, False]

result = {
    'grid_res': [],
    'speedup': [],
    'RMSE': [],
    'device': [],
}


def experiment(device):
    grid_res = 50

    domain = Domain()
    domain.variable('x', [0, 1], grid_res)
    domain.variable('t', [0, 1], grid_res)

    boundaries = Conditions()
    # u(x,0)=1e4*sin^2(x(x-1)/10)
    x = domain.variable_dict['x']
    func_bnd1 = lambda x: 10 ** 4 * torch.sin((1 / 10) * x * (x - 1)) ** 2
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=func_bnd1(x))

    # du/dx (x,0) = 1e3*sin^2(x(x-1)/10)
    func_bnd2 = lambda x: 10 ** 3 * torch.sin((1 / 10) * x * (x - 1)) ** 2

    bop2 = {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            }
    }

    boundaries.operator({'x': [0, 1], 't': 0}, operator=bop2, value=func_bnd2(x))

    # u(0,t) = u(1,t)
    boundaries.periodic([{'x': 0, 't': [0, 1]}, {'x': 1, 't': [0, 1]}])

    # du/dt(0,t) = du/dt(1,t)
    bop4 = {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.periodic([{'x': 0, 't': [0, 1]}, {'x': 1, 't': [0, 1]}], operator=bop4)

    equation = Equation()

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
                'coeff': -1 / C,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(wave_eq)

    # NN
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1))

    net_1 = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1))

    start = time.time()

    model = Model(net, domain, equation, boundaries)

    model.compile("NN", lambda_operator=1, lambda_bound=1000, h=0.01)

    cb_cache = Cache(verbose=True, model_randomize_parameter=1e-5)

    cb_es = EarlyStopping(eps=1e-6,
                          loss_window=1000,
                          no_improvement_patience=500,
                          patience=10,
                          randomize_parameter=0,
                          info_string_every=500)

    img_dir = os.path.join(os.path.dirname(__file__), 'wave_eq_img')

    cb_plots = Plots(save_every=500, print_every=None, img_dir=img_dir)

    optimizer = Optimizer(model=net, optimizer_type='Adam', learning_rate=1e-2)

    model.train(optimizer=optimizer, epochs=1e5, save_model=False, mixed_precision=True, device=device,
                callbacks=[cb_es, cb_cache, cb_plots])

    end = time.time()

    start_1 = time.time()

    model1 = Model(net, domain, equation, boundaries)

    model1.compile("NN", lambda_operator=1, lambda_bound=1000, h=0.01)

    model1.train(optimizer=optimizer, epochs=1e5, save_model=False, mixed_precision=False, device=device,
                 callbacks=[cb_es, cb_cache, cb_plots])

    end_1 = time.time()

    net = net.cpu()
    net_1 = net_1.cpu()

    grid = domain.build('NN').cpu()

    mp_true = net(grid).detach().cpu().numpy().flatten()
    mp_false = net(grid).detach().cpu().numpy().flatten()

    rmse = np.mean(np.square(mp_true - mp_false))

    result['grid_res'].append(grid_res)
    result['speedup'].append((end_1 - start_1) / (end - start))
    result['RMSE'].append(rmse)
    result['device'].append(device)
    print('Time taken = ', end - start)


for _ in range(10):
    experiment(devices[1])

df = pd.DataFrame(result)

df.to_csv('examples/benchmarking_data/wave_exp_AMP_speedup.csv', index=False)
