import torch
from typing import Union, List

from tedeous.callbacks import Callback
from tedeous.data import Domain, Conditions, Equation
from tedeous.input_preprocessing import InitialDataProcessor, lambda_prepare
from tedeous.eval import Operator, Bounds
from tedeous.points_type import Points_type
from tedeous.utils import create_random_fn

from tedeous.callbacks import CallbackList
from tedeous.device import check_device, solver_device, device_type
from tedeous.solution import Solution
from tedeous.optimizers import OptimizerStep, Optimizer

class Model():
    """class for preprocessing"""

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

    def _compile_nn(self, **params):
        self.h = params.get('h', None)
        self.inner_order = params.get('inner_order', '1')
        self.boundary_order = params.get('boundary_order', '2')
        sorted_grid = Points_type(self.grid).grid_sort()
        self.n_t = len(sorted_grid['central'][:, 0].unique())

    def _compile_mat(self, **params):
        self.derivative_points = params.get('derivative_points', None)
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

    def _prepare_lambda(self, lambda_):
        with torch.no_grad():
            op = self.operator.operator_compute()
            bval, _, _, _ = self.boundary.apply_bcs()
        self.lambda_operator = lambda_prepare(op, lambda_[0])
        self.lambda_bound = lambda_prepare(bval, lambda_[1])

    def _loss_parameters(self, **params):
        self.tol = params.get('tol', 0)
        self.weak_form = params.get('weak_form', None)

    def _misc_parameters(self, **params):
        model_randomize_parameter = params.get('model_randomize_parameter', 0.01)

        self._r = create_random_fn(model_randomize_parameter)
        self.save_graph = params.get('save_graph', False)
        self.lambda_operator = params.get('lambda_operator', 1)
        self.lambda_bound = params.get('lambda_bound', 100)
        self.normalized_loss_stop = params.get('normalized_loss_stop', False)

    def _set_solver_device(self, device):
        solver_device(device)
        self.grid = check_device(self.grid)
        self.net = self.net.to(device_type())

    def compile(
            self,
            mode: str = 'autograd',
            **params):
        """
        
        Args:
            mode: 


        Returns:

        """
        self._loss_parameters(**params)
        self._misc_parameters(**params)

        if mode == 'NN':
            self._compile_nn(**params)
        elif mode == 'autograd':
            self._compile_autograd(**params)
        elif mode == 'mat':
            self._compile_mat(**params)

        self.grid = self.domain.build(mode=mode)
        variable_dict = self.domain.variable_dict
        operator = self.equation.equation_lst
        bconds = self.conditions.build(variable_dict)

        equation_cls = InitialDataProcessor(
            self.grid,
            operator,
            bconds,
            h=self.h,
            inner_order=self.inner_order,
            boundary_order=self.boundary_order).set_strategy(mode)

        self.solution_cls = Solution(self.grid, equation_cls, self.net, mode, self.weak_form,
                                     self.lambda_operator, self.lambda_bound, self.tol, self.derivative_points)


    def train(self,
              optimizer: Optimizer,
              callbacks: List[Callback],
              epochs: int = 10000,
              verbose: int = 0 ,
              print_every: Union[None, int] = None,
              device: str = 'cpu',
              mixed_precision: bool = False):

        self.optimizer = optimizer.set_optimizer()
        self._set_solver_device(device)

        opt_step = OptimizerStep(mixed_precision, self)
        closure = opt_step.step()

        callbacks = CallbackList(callbacks=callbacks, verbose=verbose, print_every=print_every, model=self)
        callbacks.on_train_begin()

        self.t = 0
        self.stop_training = False

        while self.t < epochs:
            callbacks.on_epoch_begin()

            self.optimizer.step(closure)

            callbacks.on_epoch_end()

            if self.stop_training:
                break

        callbacks.on_train_end()