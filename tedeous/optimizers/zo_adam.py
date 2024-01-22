import torch
from typing import Callable, Tuple


class ZO_AdaMM(torch.optim.Optimizer):
    """
    Zeroth-order Adam optimizer.

    References:
        - Xiangyi Chen, Sijia Liu, Kaidi Xu, Xingguo Li, Xue Lin, Mingyi Hong, David Cox.
          "ZO-AdaMM: Zeroth-Order Adaptive Momentum Method for Black-Box Optimization."
          ArXiv preprint, abs/1910.06513, 2019.
          URL: https://api.semanticscholar.org/CorpusID:202777327
    """

    def __init__(self, params: torch.Tensor,
                 input_size: int,
                 gradient_mode: str = 'forward',
                 sampler: str = 'uniform',
                 dim: int = 2,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 mu: float = 1e-3,
                 eps: float = 1e-12):
        """
        Args:
            params:
            input_size: size of the input data (i.e. batch size).
            gradient_mode: mode for gradient descent directions (e.g. 'central').
            sampler: random sampling type (uniform, normal).
            dim: problem dimensionality (d = 1 - ODEs, d > 2 - PDEs i.e. t, x, y, z, etc...).
            lr: learning rate.
            betas: coefficients used for computing running averages of gradient and its square.
            mu: perturbation parameter for each direction in gradient size (i.e. standard deviation).
            eps: term added to the denominator to improve numerical stability.
        """
        defaults = dict(lr=lr, betas=betas, mu=mu, eps=eps)
        super().__init__(params, defaults)
        self.input_size = input_size
        self.gradient_mode = gradient_mode
        self.sampler = sampler
        self.n_samples = 1
        self.dim = dim
        self.name = 'ZO_Adam'

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
            beta1, beta2 = group['betas']

            # Closure return the approximation for the gradient
            grad_est, loss = closure(self.size_params, group["mu"],
                               self.n_samples, self.input_size,
                               self.dim, self.sampler, self.gradient_mode)

            for p, grad in zip(group['params'], grad_est):
                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Do the AdaMM updates
                state['exp_avg'].mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))
                state['max_exp_avg_sq'] = torch.maximum(state['max_exp_avg_sq'],
                                                        state['exp_avg_sq'])

                p.data.addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(group['eps']), value=(-group['lr']))

        return loss