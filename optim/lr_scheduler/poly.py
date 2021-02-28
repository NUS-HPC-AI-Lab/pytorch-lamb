from torch.optim.lr_scheduler import _LRScheduler


class PolynomialDecay(_LRScheduler):
    def __init__(self, optimizer, decay_steps, end_lr=0.0001, power=1.0, last_epoch=-1):
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        return [
            (base_lr - self.end_lr) * ((1 - min(self.last_epoch, self.decay_steps) /
                                        self.decay_steps) ** self.power) + self.end_lr
            for base_lr in self.base_lrs
        ]
