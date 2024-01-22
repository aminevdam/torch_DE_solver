from torch.optim.lr_scheduler import ExponentialLR
from tedeous.callbacks import Callback


class LRScheduler(Callback):
    """
    Updates the learning rate after each given iteration.
    """

    def __init__(self, gamma, decay_rate):
        super().__init__()
        self.decay_rate = decay_rate
        optimizer = self.model.optimizer
        self.scheduler = ExponentialLR(optimizer, gamma)

    def on_epoch_end(self, logs=None):
        if self.model.t % self.decay_rate == 0:
            self.scheduler.step()
