import torch
from abc import ABC
from typing import Union
from tedeous.optimizers import PSO, ZO_AdaMM, ZO_SignSGD

class Optimizer():
    def __init__(self, model, optimizer_type: str, learning_rate: float = 1e-3, **params):
        self.model = model
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.mode = params.get('mode', 'NN')
        self.params = params

    def _optimizer_choice(self):
        """ Setting optimizer. If optimizer is string type, it will get default settings,
            or it may be custom optimizer defined by user.

        Args:
           optimizer: optimizer choice (Adam, SGD, LBFGS, PSO).
           learning_rate: determines the step size at each iteration
           while moving toward a minimum of a loss function.

        Returns:
            optimizer: ready optimizer.
        """
        if self.optimizer_type == 'Adam':
            torch_optim = torch.optim.Adam
        elif self.optimizer_type == 'SGD':
            torch_optim = torch.optim.SGD
        elif self.optimizer_type == 'LBFGS':
            torch_optim = torch.optim.LBFGS
        elif self.optimizer_type == 'PSO':
            torch_optim = PSO
        elif self.optimizer_type == 'ZO_Adam':
            torch_optim = ZO_AdaMM
        elif self.optimizer_type == 'ZO_SignSGD':
            torch_optim = ZO_SignSGD

        return torch_optim

    def set_optimizer(self):
        optimizer = self._optimizer_choice()
        if self.mode in ('NN', 'autograd'):
            optimizer = optimizer(self.model.parameters(), lr=self.learning_rate, **self.params)
        elif self.mode == 'mat':
            optimizer = optimizer([self.model.requires_grad_()], lr=self.learning_rate, **self.params)
        return optimizer