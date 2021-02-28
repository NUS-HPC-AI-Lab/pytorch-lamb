from torch.optim.lr_scheduler import OneCycleLR

__all__ = [
    'OneCycle'
]


class OneCycle(OneCycleLR):
    def __init__(self, optimizer, decay_steps,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.0,
                 final_div_factor=10000.0,
                 last_epoch=-1):
        max_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))
        super().__init__(optimizer, max_lrs, total_steps=decay_steps,
                         pct_start=pct_start,
                         anneal_strategy=anneal_strategy,
                         cycle_momentum=cycle_momentum,
                         base_momentum=base_momentum,
                         max_momentum=max_momentum,
                         div_factor=div_factor,
                         final_div_factor=final_div_factor,
                         last_epoch=last_epoch)
