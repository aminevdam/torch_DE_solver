from torch.optim.lr_scheduler import ExponentialLR
from tedeous.callbacks import Callback


class LRScheduler(Callback):
    """
    Updates the learning rate after each given iteration.
    """

    def __init__(self, gamma, decay_rate):
        super().__init__()
        self.gamma = gamma
        self.decay_rate = decay_rate

    def on_epoch_end(self, logs=None):
        scheduler = ExponentialLR(self.model.optimizer, self.gamma)
        if self.model.t % self.decay_rate == 0:
            scheduler.step()
            print(f'Current learning rate: {scheduler.get_last_lr()[0]}')
