import torch
import numpy as np

from typing import Any
from copy import deepcopy

from tedeous.device import check_device

class OptimizerStep:
    def __init__(self,
        mixed_precision: bool,
        model,
        # dtype: Any,
        # second_order_interactions: bool,
        # sampling_N: int,
        # lambda_update: bool,
        # normalized_loss_stop: bool,
                 **params):


        self.mixed_precision = mixed_precision
        self.second_order_interactions = params.get('second_order_interactions', True)
        self.sampling_N = params.get('sampling_N', 1)

        self.set_model(model)
        self.optimizer = self.model.optimizer
        self.normalized_loss_stop = self.model.normalized_loss_stop

    def set_model(self, model):
        self._model = model

    @property
    def model(self):
        return self._model
    def _amp_mixed(self, mixed_precision: bool):
        """ Preparation for mixed precsion operations.

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

    def _closure_default(self):
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.model.device, dtype=self.dtype, enabled=self.mixed_precision):
            op = self.model.operator.operator_compute()
            bval, true_bval, bval_keys, bval_length = self.model.boundary.apply_bcs()

            loss, loss_normalized = self.model.loss_cls.compute(op, bval, true_bval,
                                                                self.model.lambda_operator,
                                                                self.model.lambda_bound,
                                                                self.model.save_graph)

        loss.backward()
        self.cur_loss = loss_normalized if self.normalized_loss_stop else loss
        return loss

    def _closure_cuda(self):
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.model.device, dtype=self.dtype, enabled=self.mixed_precision):
            op = self.model.operator.operator_compute()
            bval, true_bval, bval_keys, bval_length = self.model.boundary.apply_bcs()

            loss, loss_normalized = self.model.loss_cls.compute(op, bval, true_bval,
                                                          self.model.lambda_operator,
                                                          self.model.lambda_bound,
                                                          self.model.save_graph)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.cur_loss = loss_normalized if self.normalized_loss_stop else loss
        return loss

    def _closure_zo(self, size_params, mu, N_samples, input_size, d, sampler, gradient_mode):
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
        self.cur_loss, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)

        for _ in range(N_samples):
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
            loss_add, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)

            # param - mu * eps
            with torch.no_grad():
                parameter_perturbation(-2 * u)
            loss_sub, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)

            with torch.no_grad():
                if gradient_mode == 'central':
                    # (1/ inp_size * q) * d * [f(x+mu*eps) - f(x-mu*eps)] / 2*mu
                    grad_coeff = (1 / (input_size * N_samples)) * d * (loss_add - loss_sub) / (2 * mu)
                elif gradient_mode == 'forward':
                    # d * [f(x+mu*eps) - f(x)] / mu
                    grad_coeff = (1 / (input_size * N_samples)) * d * (loss_add - self.cur_loss) / mu
                elif gradient_mode == 'backward':
                    # d * [f(x) - f(x-mu*eps)] / mu
                    grad_coeff = (1 / (input_size * N_samples)) * d * (self.cur_loss - loss_sub) / mu

                # coeff * u, i.e. constant multiplied by infinitely small perturbation.
                current_grad = grads_multiplication(grad_coeff, u)

                grads = [grad_past + cur_grad for grad_past, cur_grad in zip(grads, current_grad)]

            # load initial model parameters
            self.model.load_state_dict(init_model_parameters)

            loss_checker, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)
            assert self.cur_loss == loss_checker

        return grads