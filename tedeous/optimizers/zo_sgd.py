import torch
from typing import Callable

class ZO_SignSGD(torch.optim.Optimizer):
    """
    Zeroth-order Sign Stochastic Gradient Descent optimizer.

    References:
        - Sijia Liu, Pin-Yu Chen, Xiangyi Chen, Mingyi Hong.
          "Sign{SGD} via Zeroth-Order Oracle."
          International Conference on Learning Representations, 2019.
          URL: https://openreview.net/forum?id=BJe-DsC5Fm
    """
    def __init__(self, params,
                 input_size,
                 gradient_mode='central',
                 sampler='normal',
                 n_samples=5,
                 dim=2,
                 lr=1e-3,
                 mu=1e-3):
        """
        Args:
            params: model parameters.
            input_size: size of the input data (i.e. batch size).
            gradient_mode: mode for gradient descent directions (e.g. 'central').
            sampler: random sampling type (uniform, normal).
            n_samples: set size for gradient descent directions.
            dim: problem dimensionality (d = 1 - ODEs, d > 2 - PDEs i.e. t, x, y, z, etc...).
            lr: learning rate.
            mu: perturbation parameter for each direction in gradient size (i.e. standard deviation).
        """
        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)
        self.input_size = input_size
        self.gradient_mode = gradient_mode
        self.sampler = sampler
        self.n_samples = n_samples
        self.dim = dim
        self.name = 'ZO_SignSGD'

        self.size_params = 0
        for group in self.param_groups:
            for p in group['params']:
                self.size_params += torch.numel(p)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Args:
            closure: function that computes gradients.
        """
        for group in self.param_groups:
            lr = group['lr']
            for i, param in enumerate(group['params']):
                grad_est, loss = closure(self.size_params, group["mu"],
                                   self.n_samples, self.input_size,
                                   self.dim, self.sampler, self.gradient_mode)

                param.data.add_(-lr * torch.sign(grad_est[i]))
        return loss