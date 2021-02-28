from torch.optim.lr_scheduler import MultiStepLR


class MultiStep(MultiStepLR):
    def __init__(self, optimizer, decay_steps, milestones=None, gamma=0.1, last_epoch=-1):
        super().__init__(optimizer, milestones, gamma=gamma, last_epoch=last_epoch)
