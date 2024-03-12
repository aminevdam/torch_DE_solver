import torch
import numpy as np
from typing import Tuple
from copy import deepcopy

from tedeous.device import check_device


class OptimizerStep:
    """
    Setting optimizer step function (i.e. closure in terms of pytorch).
    """
    def __init__(self,
                 mixed_precision: bool,
                 model):
        self.set_model(model)
        self.optimizer = self.model.optimizer
        self.normalized_loss_stop = self.model.normalized_loss_stop
        self.mixed_precision = mixed_precision

    def set_model(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    def _amp_mixed(self, mixed_precision: bool):
        """
        Preparation for mixed precision operations.

        Args:
            mixed_precision (bool): use or not torch.amp.

        Raises:
            NotImplementedError: AMP and the LBFGS optimizer are not compatible.

        Returns:
            scaler: GradScaler for CUDA.
            cuda_flag (bool): True, if CUDA is activated and mixed_precision=True.
            dtype (dtype): operations dtype.
        """

        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        print(f'Mixed precision enabled. The device is {self.model.device}')
        if self.optimizer.__class__.__name__ == "LBFGS":
            raise NotImplementedError("AMP and the LBFGS optimizer are not compatible.")
        self.cuda_flag = True if self.model.device == 'cuda' and mixed_precision else False
        self.dtype = torch.float16 if self.model.device == 'cuda' else torch.bfloat16

    def _closure_cpu(self):
        """
        Default pytorch closure function. Support CPU mixed_precision.
        """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.model.device, dtype=self.model.dtype, enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()

        loss.backward()
        self.cur_loss = loss_normalized if self.model.normalized_loss_stop else loss
        return loss

    def _closure_default(self):
        """
        Default pytorch closure function. Support CPU mixed_precision.
        """
        self.optimizer.zero_grad()

        loss, loss_normalized = self.model.solution_cls.evaluate()

        loss.backward()

        self.cur_loss = loss_normalized if self.model.normalized_loss_stop else loss
        return loss

    def _closure_cuda(self):
        """
        Closure function for CUDA. Support CUDA mixed_precision.
        """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.model.device, dtype=self.dtype, enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.cur_loss = loss_normalized if self.normalized_loss_stop else loss
        return loss

    def _closure_zo(self,
                    size_params: int,
                    input_size: int,
                    mu: float,
                    n_samples: int,
                    d: int,
                    sampler: str = 'uniform',
                    gradient_mode: str = 'central') -> Tuple[list, torch.Tensor]:
        """
        Closure function for zeroth-order optimizers.

        Args:
            size_params: number of optimization parameters.
                            The calculation occurs automatically. Do not set it manually!
            input_size: size of the input data (i.e. batch size).
            n_samples: set size for gradient descent directions.
            mu: perturbation parameter for each direction in gradient size (i.e. standard deviation).
            d: problem dimensionality (d = 1 - ODEs, d > 2 - PDEs i.e. [t, x, y, z, etc...]).
            sampler: random sampling type.
            gradient_mode: mode for gradient descent directions.

        Returns:
            zeroth-order gradient estimation.
        """
        init_model_parameters = deepcopy(dict(self.model.net.state_dict()))
        model_parameters = dict(self.model.net.state_dict()).values()

        def parameter_perturbation(eps):
            start_idx = 0
            for param in model_parameters:
                end_idx = start_idx + param.view(-1).size()[0]
                param.add_(eps[start_idx: end_idx].view(param.size()).float(), alpha=np.sqrt(mu))
                start_idx = end_idx

        def grads_multiplication(grads, u):
            start_idx = 0
            grad_est = []
            for param in model_parameters:
                end_idx = start_idx + param.view(-1).size()[0]
                grad_est.append(grads * u[start_idx:end_idx].view(param.size()))
                start_idx = end_idx
            return grad_est

        grads = [torch.zeros_like(param) for param in model_parameters]
        self.cur_loss, _ = self.model.solution_cls.evaluate()

        for _ in range(n_samples):
            with torch.no_grad():
                if sampler == 'uniform':
                    u = 2 * (torch.rand(size_params) - 0.5)
                    u.div_(torch.norm(u, "fro"))
                    u = check_device(u)
                elif sampler == 'normal':
                    u = torch.randn(size_params)
                    u = check_device(u)

                # param + mu * eps
                parameter_perturbation(u)

            loss_add, _ = self.model.solution_cls.evaluate()

            # param - mu * eps
            with torch.no_grad():
                parameter_perturbation(-2 * u)

            loss_sub, _ = self.model.solution_cls.evaluate()

            with torch.no_grad():
                if gradient_mode == 'central':
                    # (1/ inp_size * q) * d * [f(x+mu*eps) - f(x-mu*eps)] / 2*mu
                    grad_coeff = (1 / (input_size * n_samples)) * d * (loss_add - loss_sub) / (2 * mu)
                elif gradient_mode == 'forward':
                    # d * [f(x+mu*eps) - f(x)] / mu
                    grad_coeff = (1 / (input_size * n_samples)) * d * (loss_add - self.cur_loss) / mu
                elif gradient_mode == 'backward':
                    # d * [f(x) - f(x-mu*eps)] / mu
                    grad_coeff = (1 / (input_size * n_samples)) * d * (self.cur_loss - loss_sub) / mu

                # coeff * u, i.e. constant multiplied by infinitely small perturbation.
                current_grad = grads_multiplication(grad_coeff, u)

                grads = [grad_past + cur_grad for grad_past, cur_grad in zip(grads, current_grad)]

            # load initial model parameters
            self.model.net.load_state_dict(init_model_parameters)

            loss, loss_norm = self.model.solution_cls.evaluate()
            assert self.cur_loss == loss

        return grads, self.cur_loss

    def step(self):
        """
        Defines optimizer step.

        Returns:
            closure function.
        """
        if self.mixed_precision:
            return self._closure_cuda if self.model.device == 'cuda' else self._closure_cpu
        elif self.optimizer.__class__.__name__[:2] == 'ZO':
            return self._closure_zo
        return self._closure_default
