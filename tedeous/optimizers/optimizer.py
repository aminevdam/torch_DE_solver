import torch
from abc import ABC
from typing import Union, Any
from tedeous.optimizers.pso import PSO
from torch.optim.lr_scheduler import ExponentialLR


class Optimizer:
    """
    Setting the optimizer for the model.
    """
    def __init__(self,
                 optimizer_type: str = 'Adam',
                 learning_rate: float = 1e-3,
                 **params):
        """
        Args:
            model: model.
            optimizer_type: optimizer type.
            learning_rate: determines the step size at each iteration
            while moving toward a minimum of a loss function.
            **params: additional parameters for the optimizer (e.g. ZO parameters, beta parameters for Adam).
        """
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.params = params

    def set_optimizer(
            self,
            mode,
            model) -> \
            Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS, PSO]:
        """ Setting optimizer. If optimizer is string type, it will get default settings,
            or it may be custom optimizer defined by user.

        Args:
           optimizer: optimizer choice (Adam, SGD, LBFGS, PSO).
           learning_rate: determines the step size at each iteration
           while moving toward a minimum of a loss function.

        Returns:
            optimzer: ready optimizer.
        """
        if not isinstance(self.optimizer_type, str):
            return self.optimizer_type

        if self.optimizer_type == 'Adam':
            torch_optim = torch.optim.Adam
        elif self.optimizer_type == 'SGD':
            torch_optim = torch.optim.SGD
        elif self.optimizer_type == 'LBFGS':
            torch_optim = torch.optim.LBFGS
        elif self.optimizer_type == 'PSO':
            torch_optim = PSO

        if mode in ('NN', 'autograd'):
            optimizer = torch_optim(model.parameters(), **self.params)
        elif mode == 'mat':
            optimizer = torch_optim([model.requires_grad_()], **self.params)

        return optimizer