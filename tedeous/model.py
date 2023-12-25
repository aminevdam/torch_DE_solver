import torch

from typing import Union

from tedeous.input_preprocessing import Equation
from tedeous.points_type import Points_type
from tedeous.eval import Operator, Bounds
from tedeous.losses import Losses
from tedeous.device import device_type, check_device
from tedeous.input_preprocessing import lambda_prepare
from tedeous.utils import create_random_fn
from tedeous.callbacks.callback_list import CallbackList


class Model:
    def __init__(self,
                 model,
                 grid,
                 equation,
                 boundary_conditions):

        self.model = model.to(device_type())
        self.grid = check_device(grid)
        self.equation = equation  # aka new cls for input data
        self.bconds = boundary_conditions

    @staticmethod
    def __operator_coeff(equal_cls, operator):
        for i in range(len(operator)):
            eq = operator[i]
            for key in eq.keys():
                if isinstance(eq[key]['coeff'], torch.Tensor):
                    try:
                        eq[key]['coeff'] = equal_cls.operator[i][key]['coeff'].to(device_type())
                    except:
                        eq[key]['coeff'] = equal_cls.operator[key]['coeff'].to(device_type())


    def compile(self,
                mode: str = 'NN',
                weak_form: Union[None, list] = None,
                lambda_op: int = 1,
                lambda_bcs: int = 100,
                **kwargs):
        """

        """
        # Optional parameters
        self.mode = mode
        model_randomize_parameter = kwargs.get('model_randomize_parameter', 0.001)
        self._r = create_random_fn(model_randomize_parameter)

        if mode == 'NN':
            sorted_grid = Points_type(self.grid).grid_sort()
            n_t = len(sorted_grid['central'][:, 0].unique())
        elif mode == 'autograd':
            n_t = len(self.grid[:, 0].unique())
        elif mode == 'mat':
            n_t = self.grid.shape[1]

        # Mode NN
        h = kwargs.get('h', 0.001)
        inner_order = kwargs.get('inner_order', '1')
        boundary_order = kwargs.get('boundary_order', '2')

        # Mode Mat
        derivative_points = kwargs.get('derivative_points', 2)

        # Loss parameters
        tol = kwargs.get('tol', 0.)
        self.save_graph = kwargs.get('save_graph', True)

        self.equation = Equation(self.grid, self.equation, self.bconds, h=h).set_strategy(mode)

        prepared_operator = self.equation.operator_prepare()
        prepared_bconds = self.equation.bnd_prepare()

        self.__operator_coeff(self.equation, prepared_operator)

        self.operator = Operator(self.grid, prepared_operator, self.model,
                                 mode, weak_form, derivative_points)
        self.boundary = Bounds(self.grid, prepared_bconds, self.model,
                               mode, weak_form, derivative_points)

        self.loss_cls = Losses(mode, weak_form, n_t, tol)

        self.lambda_operator = lambda_prepare(len(prepared_operator), lambda_op)
        self.lambda_bound = lambda_prepare(len(prepared_bconds), lambda_bcs)

    def train(self, callbacks,
              optimizer,
              epochs: int = 20000,
              verbose: int = 0,
              print_every: Union[int, None] = None,
              device: str = 'cpu',
              mixed_precision: bool = False):

        op = self.operator.operator_compute()
        bval, true_bval, bval_keys, bval_length = self.boundary.apply_bcs()

        self.min_loss = self.loss_cls.compute(op, bval, true_bval,
                                                          self.lambda_operator,
                                                          self.lambda_bound,
                                                          self.save_graph)

        callbacks = CallbackList(callbacks=callbacks, verbose=verbose, print_every=print_every, model=self)
        callbacks.on_train_begin()

        self.t = 0
        self.stop_training = False

        while self.t < epochs:
            callbacks.on_epoch_begin()

            op = self.operator.operator_compute()

            bval, true_bval, bval_keys, bval_length = self.boundary.apply_bcs()
            loss, loss_normalized = self.loss_cls.compute(op, bval, true_bval,
                                                          self.lambda_operator,
                                                          self.lambda_bound,
                                                          self.save_graph)

            optimizer.step(self.closure)

            self.t += 1
            callbacks.on_epoch_end()

            if self.stop_training:
                break

