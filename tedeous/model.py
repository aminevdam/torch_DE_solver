import torch
from typing import Union, List
import datetime
import time

from tedeous.data import Domain, Conditions, Equation
from tedeous.input_preprocessing import InitialDataProcessor, lambda_prepare
from tedeous.eval import Operator, Bounds
from tedeous.points_type import Points_type
from tedeous.utils import create_random_fn, CacheUtils
from tedeous.callbacks import Callback, CallbackList
from tedeous.device import check_device, solver_device, device_type
from tedeous.solution import Solution
from tedeous.optimizers import OptimizerStep, Optimizer


class Model:
    """The model is an interface for solving equations"""

    def __init__(
            self,
            net: Union[torch.nn.Module, torch.Tensor],
            domain: Domain,
            equation: Equation,
            conditions: Conditions):
        """

        Args:
            net (Union[torch.nn.Module, torch.Tensor]): neural network or torch.Tensor for mode *mat*
            grid (Domain): object of class Domain
            equation (Equation): object of class Equation
            conditions (Conditions): object of class Conditions
        """
        self.net = net
        self.domain = domain
        self.equation = equation
        self.conditions = conditions
        self._check = None

    def _compile_nn(self, **params):
        self.inner_order = params.get('inner_order', '1')
        self.boundary_order = params.get('boundary_order', '2')
        sorted_grid = Points_type(self.grid).grid_sort()
        self.n_t = len(sorted_grid['central'][:, 0].unique())

    def _compile_mat(self, **params):
        self.n_t = self.grid.shape[1]

    def _compile_autograd(self, **params):
        self.n_t = len(self.grid[:, 0].unique())

    def _prepare_operator(self, equation_cls, mode, weak_form):
        prepared_op = equation_cls.operator_prepare()
        self.operator = Operator(self.grid, prepared_op, self.net,
                                 mode, weak_form, self.derivative_points)

    def _prepare_boundary(self, equation_cls, mode, weak_form):
        prepared_bcs = equation_cls.bnd_prepare()

        self.boundary = Bounds(self.grid, prepared_bcs, self.net,
                               mode, weak_form, self.derivative_points)

    def _prepare_lambda(self, lambda_: tuple):
        """
        Prepares the lambdas to solver form.

        Args:
            lambda_: tuple of lambdas, where [0] position corresponds to the operator and
                        [1] corresponds to the boundary conditions.

        """
        with torch.no_grad():
            op = self.operator.operator_compute()
            bval, _, _, _ = self.boundary.apply_bcs()
        self.lambda_operator = lambda_prepare(op, lambda_[0])
        self.lambda_bound = lambda_prepare(bval, lambda_[1])

    def _loss_parameters(self, **params):
        """
        Configures loss function parameters.

        Args:
            **params: tol (causal loss), weak_form (weak loss)

        """
        self.tol = params.get('tol', 0)
        self.weak_form = params.get('weak_form', None)

    def _misc_parameters(self, **params):
        """
        Configures miscellaneous parameters.

        Args:
            **params: lambda_operator, lambda_bound, normalized_loss_stop.
        """
        model_randomize_parameter = params.get('model_randomize_parameter', 0.01)

        self.h = params.get('h', None)
        self._r = create_random_fn(model_randomize_parameter)
        self.save_graph = params.get('save_graph', False)
        self.lambda_operator = params.get('lambda_operator', 1)
        self.lambda_bound = params.get('lambda_bound', 100)
        self.normalized_loss_stop = params.get('normalized_loss_stop', False)
        self.inner_order = params.get('inner_order', '1')
        self.boundary_order = params.get('boundary_order', '2')
        self.derivative_points = params.get('derivative_points', None)
        self.dtype = torch.float32

    def _set_solver_device(self, device: str):
        """
        Method that sets device to 'cpu' or 'cuda'.
        Args:
            device: device.
        """

        solver_device(device)
        self.grid = check_device(self.grid)
        self.net = self.net.to(device_type())
        self.device = device

    def compile(
            self,
            mode: str = 'autograd',
            **params):
        """
        Configures the model.

        Args:
            mode: Calculation method. (e.g., "NN", "autograd", "mat").
        """
        print('Compiling model...')
        self._loss_parameters(**params)
        self._misc_parameters(**params)
        self.mode = mode

        self.grid = self.domain.build(mode=mode)
        variable_dict = self.domain.variable_dict
        operator = self.equation.equation_lst
        bconds = self.conditions.build(variable_dict)

        if mode == 'NN':
            self._compile_nn(**params)
        elif mode == 'autograd':
            self._compile_autograd(**params)
        elif mode == 'mat':
            self._compile_mat(**params)

        equation_cls = InitialDataProcessor(
            self.grid,
            operator,
            bconds,
            h=self.h,
            inner_order=self.inner_order,
            boundary_order=self.boundary_order).set_strategy(mode)

        self.solution_cls = Solution(self.grid, equation_cls, self.net, mode, self.weak_form,
                                     self.lambda_operator, self.lambda_bound, self.tol, self.derivative_points)
        print('Model compiled.')

    def _model_save(
            self,
            save_model: bool,
            model_name: str):
        """
        Model saving.

        Args:
            save_model: flag for model saving.
            model_name: name of the model.
        """
        if save_model:
            if self.mode == 'mat':
                CacheUtils().save_model_mat(model=self.net,
                                            domain=self.domain,
                                            name=model_name)
            else:
                CacheUtils().save_model(model=self.net, name=model_name)

    def train(self,
              optimizer: Optimizer,
              callbacks: List[Callback],
              epochs: Union[int, float] = 10000,
              verbose: int = 0,
              device: str = 'cpu',
              save_model: bool = False,
              model_name: Union[str, None] = None,
              print_every: Union[int, None] = None,
              mixed_precision: bool = False) -> Union[torch.nn.Module, torch.nn.Sequential]:
        """
        Trains the model.

        Args:
            print_every:
            optimizer: optimizer for training the net.
            callbacks: list of callbacks used for training.
            epochs: number of epochs to train.
            verbose: verbosity level.
            device: device to use.
            save_model: whether to save model.
            model_name: model name.
            mixed_precision: mixed precision (fp16/fp32).

        Returns:
            the trained model.
        """
        self.optimizer = optimizer.set_optimizer()
        self._set_solver_device(device)

        opt_step = OptimizerStep(mixed_precision, self)
        closure = opt_step.step()

        callbacks = CallbackList(callbacks=callbacks, verbose=verbose, model=self)
        callbacks.on_train_begin()

        self.t = 0
        self.stop_training = False

        loss, loss_norm = self.solution_cls.evaluate()
        self.min_loss = loss_norm if self.normalized_loss_stop else loss

        print('[{}] initial (min) loss is {}'.format(
            datetime.datetime.now(), self.min_loss.item()))

        start = time.time()

        while self.t < epochs:
            callbacks.on_epoch_begin()

            self.cur_loss = self.optimizer.step(closure)

            callbacks.on_epoch_end()

            if print_every is not None:
                if self.t % print_every == 0:
                    loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                    info = 'Step = {} loss = {:.6f}.'.format(self.t, loss)
                    print(info)
            if self.stop_training:
                break

            self.t += 1

        callbacks.on_train_end()

        self._model_save(save_model, model_name)

        end = time.time()
        print(f'[{end - start}] training time')

        return self.net
