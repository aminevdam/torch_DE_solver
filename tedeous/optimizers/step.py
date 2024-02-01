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

    def _amp_mixed(self):
        """
        Preparation for mixed precision operations.

        Raises:
            NotImplementedError: AMP and the LBFGS optimizer are not compatible.

        Returns:
            scaler: GradScaler for CUDA.
            cuda_flag (bool): True, if CUDA is activated and mixed_precision=True.
            dtype (dtype): operations dtype.
        """

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        print(f'Mixed precision enabled. The device is {self.model.device}')
        if self.optimizer.__class__.__name__ == "LBFGS":
            raise NotImplementedError("AMP and the LBFGS optimizer are not compatible.")
        self.cuda_flag = True if self.model.device == 'cuda' and self.mixed_precision else False
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

    def step(self):
        """
        Defines optimizer step.

        Returns:
            closure function.
        """
        if self.mixed_precision:
            self._amp_mixed()
            return self._closure_cuda if self.model.device == 'cuda' else self._closure_cpu
        return self._closure_default
