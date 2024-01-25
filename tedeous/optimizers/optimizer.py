import torch
from typing import Union
from tedeous.optimizers.pso import PSO

class Optimizer:
    """
    Setting the optimizer for the model.
    """
    def __init__(self,
                 model: Union[torch.nn.Sequential, torch.nn.Module, torch.Tensor],
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
        self.model = model
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.params = params

    def _optimizer_choice(self):
        """
        Managing the optimizer choice.

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

        return torch_optim

    def set_optimizer(self, mode):
        """
        Setting optimizer.

       Returns:
           optimizer: ready optimizer.
       """
        optimizer = self._optimizer_choice()
        if mode in ('NN', 'autograd'):
            optimizer = optimizer(self.model.parameters(), lr=self.learning_rate, **self.params)
        elif mode == 'mat':
            optimizer = optimizer([self.model.requires_grad_()], lr=self.learning_rate, **self.params)
        return optimizer
