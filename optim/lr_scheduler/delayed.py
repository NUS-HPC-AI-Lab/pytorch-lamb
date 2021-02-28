from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, MultiStepLR
from .poly import PolynomialDecay

__all__ = [
    'FlatAnnealing',
    'CosineAnnealingWarmup',
    'FlatAnnealingWarmup'
]


class DelayerScheduler(_LRScheduler):
    """ Starts with a flat lr schedule until it reaches N epochs the applies a scheduler 
    Args:
            optimizer (Optimizer): Wrapped optimizer.
            delay_epochs: number of epochs to keep the initial lr until starting aplying the scheduler
            after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, delay_epochs, after_scheduler, last_epoch=-1):
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return self.base_lrs

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
        else:
            return super(DelayerScheduler, self).step(epoch)


class FlatAnnealing(DelayerScheduler):
    def __init__(self, optimizer, decayed_steps, pct_start=0.72, last_epoch=-1):
        flat_steps = int(decayed_steps * pct_start)
        anneal_steps = decayed_steps - flat_steps
        base_scheduler = CosineAnnealingLR(
            optimizer, anneal_steps, last_epoch=last_epoch)
        super().__init__(optimizer, flat_steps, base_scheduler, last_epoch=last_epoch)


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
        else:
            return super().step(epoch)


class WarmupDelayerScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, delay_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        self.delay_epochs = int(delay_epochs)
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs + self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()
        elif self.last_epoch >= self.warmup_epochs:
            return self.base_lrs

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
        else:
            return super().step(epoch)


class CosineAnnealingWarmup(WarmupScheduler):
    def __init__(self, optimizer, decay_steps, warmup_steps=0, eta_min=0, last_epoch=-1):
        warmup_steps = int(warmup_steps)
        base_scheduler = CosineAnnealingLR(
            optimizer, decay_steps - warmup_steps, eta_min=eta_min, last_epoch=last_epoch)
        super().__init__(optimizer, warmup_steps, base_scheduler)


class FlatAnnealingWarmup(WarmupDelayerScheduler):
    def __init__(self, optimizer, decay_steps, warmup_steps=0, pct_start=0.72, eta_min=0, last_epoch=-1):
        warmup_steps = int(warmup_steps)
        flat_steps = int((decay_steps - warmup_steps) * pct_start)
        anneal_steps = decay_steps - warmup_steps - flat_steps
        base_scheduler = CosineAnnealingLR(
            optimizer, anneal_steps, eta_min=eta_min, last_epoch=last_epoch)
        super().__init__(optimizer, warmup_steps, flat_steps, base_scheduler)


class MultiStepWarmup(WarmupScheduler):
    def __init__(self, optimizer, decay_steps, warmup_steps=0, milestones=None, gamma=0.1, last_epoch=-1):
        warmup_steps = int(warmup_steps)
        milestones = [e - warmup_steps for e in milestones]
        base_scheduler = MultiStepLR(
            optimizer, milestones, gamma=gamma, last_epoch=last_epoch)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


class PolynomialWarmup(WarmupScheduler):
    def __init__(self, optimizer, decay_steps, warmup_steps=0, end_lr=0.0001, power=1.0, last_epoch=-1):
        warmup_steps = int(warmup_steps)
        base_scheduler = PolynomialDecay(
            optimizer, decay_steps - warmup_steps, end_lr=end_lr, power=power, last_epoch=last_epoch)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)
