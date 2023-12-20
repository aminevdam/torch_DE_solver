import numpy as np
from tedeous.callbacks.callback import Callback

class EarlyStopping(Callback):
    def __init__(self, eps, loss_window, no_improvement_patience, abs_loss, normalized_loss, verbose):
        super().__init__()
        self.eps = eps
        self.loss_window = loss_window
        self.no_improvement_patience = no_improvement_patience
        self.abs_loss = abs_loss
        self.normalized_loss = normalized_loss
        self.verbose = verbose

    def _line_create(self):
        """ Approximating last_loss list (len(last_loss)=loss_oscillation_window) by the line.

        Args:
            loss_oscillation_window (int): length of last_loss list.
        """
        self._line = np.polyfit(range(self.loss_window), self.last_loss, 1)


    def _window_check(self, eps: float, loss_oscillation_window: int):
        """ Stopping criteria. We devide angle coeff of the approximating
        line (line_create()) on current loss value and compare one with *eps*

        Args:
            eps (float): min value for stopping criteria.
            loss_oscillation_window (int): list of losses length.
        """
        if self.t % loss_oscillation_window == 0 and self._check is None:
            self._line_create(loss_oscillation_window)
            if abs(self._line[0] / self.cur_loss) < eps and self.t > 0:
                self._stop_dings += 1
                if self.mode in ('NN', 'autograd'):
                    self.model.apply(self._r)
                self._check = 'window_check'

    def _patience_check(self, no_improvement_patience: int):
        """ Stopping criteria. We control the minimum loss and count steps
        when the current loss is bigger then min_loss. If these steps equal to
        no_improvement_patience parameter, the stopping criteria will be achieved.

        Args:
            no_improvement_patience (int): no improvement steps param.
        """
        if (self.t - self._t_imp_start) == no_improvement_patience and self._check is None:
            self._t_imp_start = self.t
            self._stop_dings += 1
            if self.mode in ('NN', 'autograd'):
                self.model.apply(self._r)
            self._check = 'patience_check'

    def _absloss_check(self, abs_loss: float):
        """ Stopping criteria. If current loss absolute value is lower then *abs_loss* param,
        the stopping criteria will be achieved.

        Args:
            abs_loss (float): stopping parameter.
        """
        if abs_loss is not None and self.cur_loss < abs_loss and self._check is None:
            self._stop_dings += 1

            self._check = 'absloss_check'