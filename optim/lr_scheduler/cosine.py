from torch.optim.lr_scheduler import CosineAnnealingLR

__all__ = [
    'CosineAnnealing'
]


class CosineAnnealing(CosineAnnealingLR):
    def __init__(self, optimizer, decay_steps, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, decay_steps, eta_min=eta_min, last_epoch=last_epoch)
