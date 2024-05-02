import numpy as np
import torchopt
import torch
import torch.nn as nn

from t3nsor.layers import TTLinear

class CompressedModel(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 tt_dim):
        super(CompressedModel, self).__init__()

        self.input_layer = TTLinear(input_dim, hidden_dim, d=tt_dim)
        self.hid_l1 = TTLinear(hidden_dim, hidden_dim, d=tt_dim)
        self.hid_l2 = TTLinear(hidden_dim, hidden_dim, d=tt_dim)
        self.output_layer = TTLinear(hidden_dim, output_dim)
        self.nonlinearity = nn.Tanh()

        self.model = nn.Sequential(self.input_layer,
                                   self.nonlinearity,
                                   self.hid_l1,
                                   self.nonlinearity,
                                   self.hid_l2,
                                   self.output_layer)
    def forward(self, x):
        x = self.model(x)
        return x

class CompressedZOModel(torchopt.nn.ZeroOrderGradientModule, method='forward', num_samples=100, sigma=0.01):
    def __init__(self, input_dim, hidden_dim, output_dim, tt_dim):
        super(CompressedZOModel, self).__init__()

        self.input_layer = TTLinear(input_dim, hidden_dim, d=tt_dim)
        self.hid_l1 = TTLinear(hidden_dim, hidden_dim, d=tt_dim)
        self.hid_l2 = TTLinear(hidden_dim, hidden_dim, d=tt_dim)
        self.output_layer = TTLinear(hidden_dim, output_dim)
        self.nonlinearity = nn.Tanh()
        self.distribution = torch.distributions.Normal(loc=0, scale=1)
        self.model = nn.Sequential(self.input_layer,
                                   self.nonlinearity,
                                   self.hid_l1,
                                   self.nonlinearity,
                                   self.hid_l2,
                                   self.output_layer)

    def forward(self, x):
        x = self.model(x)
        return x

    def sample(self, sample_shape=torch.Size()):
        return self.distribution.sample(sample_shape)

device = 'cuda'

torch.set_default_device(device)

############### Сетка
x_grid = np.linspace(0, 1, 51)
t_grid = np.linspace(0, 1, 51)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float().to(device)


def nn_autograd_simple(model, points, order, axis=0):
    points.requires_grad = True
    f = model(points).sum()
    for i in range(order):
        grads, = torch.autograd.grad(f, points, create_graph=True)
        f = grads[:, axis].sum()
    return grads[:, axis]



############### Граничные условия
func_bnd1 = lambda x: 10 ** 4 * torch.sin((1 / 10) * x * (x - 1)) ** 2
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float().to(device)
bndval1 = func_bnd1(bnd1[:, 0])

# du/dx (x,0) = 1e3*sin^2(x(x-1)/10)
func_bnd2 = lambda x: 10 ** 3 * torch.sin((1 / 10) * x * (x - 1)) ** 2
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float().to(device)
bop2 = {
    'du/dt':
        {
            'coeff': 1,
            'du/dt': [1],
            'pow': 1,
            'var': 0
        }
}
bndval2 = func_bnd2(bnd2[:, 0])

# u(0,t) = u(1,t)
bnd3_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float().to(device)
bnd3_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float().to(device)
bnd3 = [bnd3_left, bnd3_right]

# du/dt(0,t) = du/dt(1,t)
bnd4_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float().to(device)
bnd4_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float().to(device)
bnd4 = [bnd4_left, bnd4_right]

bop4 = {
    'du/dx':
        {
            'coeff': 1,
            'du/dx': [0],
            'pow': 1,
            'var': 0
        }
}
bcond_type = 'periodic'

bconds = [[bnd1, bndval1, 'dirichlet'],
          [bnd2, bop2, bndval2, 'operator'],
          [bnd3, bcond_type],
          [bnd4, bop4, bcond_type]]

############### ДУ + Лосс

def wave_op(model, grid):
    u_xx = nn_autograd_simple(model, grid, order=2, axis=0)
    u_tt = nn_autograd_simple(model, grid, order=2, axis=1)
    a = -(1 / 4)

    op = u_tt + a * u_xx

    return op

def op_loss(operator):
    return torch.mean(torch.square(operator))

def bcs_loss(model):
    bc1 = model(bnd1)
    bc2 = nn_autograd_simple(model, bnd2, order=1, axis=1)
    bc3 = model(bnd3_left) - model(bnd3_right)
    bc4 = nn_autograd_simple(model, bnd4_left, order=1, axis=0) - nn_autograd_simple(model, bnd4_right, order=1, axis=0)

    loss_bc1 = torch.mean(torch.square(bc1.reshape(-1) - bndval1))
    loss_bc2 = torch.mean(torch.square(bc2.reshape(-1) - bndval2))
    loss_bc3 = torch.mean(torch.square(bc3))
    loss_bc4 = torch.mean(torch.square(bc4))

    loss = loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4
    return loss

def loss_fn(model):
    operator = wave_op(model, grid)
    loss = op_loss(operator) + 1000 * bcs_loss(model)
    return loss

############### Модель
model = CompressedZOModel(2, 100, 1, 3)
n_parameters =  sum(p.numel() for p in model.parameters())
print('Number of parameters: {}'.format(n_parameters))


############### Оптимизатор
optimizer = torchopt.Adam(model.parameters(), lr=0.01)
import torch.nn.functional as F
class Net(torchopt.nn.ZeroOrderGradientModule, method='forward', num_samples=100, sigma=0.01):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.distribution = torch.distributions.Normal(loc=0, scale=1)

    def forward(self, x):
        y_pred = self.fc(x)
        # loss = F.mse_loss(y_pred, y)
        return y_pred

    def sample(self, sample_shape=torch.Size()):
        return self.distribution.sample(sample_shape)
net = Net(2)
net(grid)
# for i in range(25):
#     loss = loss_fn(model)  # compute loss
#
#     optimizer.zero_grad()
#     loss.backward()  # backward pass
#     optimizer.step()  # update network parameters
#
#     print(f'{i + 1:03d}: {loss!r}')