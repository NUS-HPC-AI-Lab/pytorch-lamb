configs = {
    'seed': 42,
    'num_epochs': 90,
    'batch_size': 64,
    'total_batch_size': 512,
    'dataset_path': '/data',
    'num_workers': 12,
    'num_threads': 12,
    'base_lr': 0.0025 / (2**0.5),
    'lr_scaling': 'sqrt',
    'weight_decay': 1.5,
    'warmup_epochs': 0.3125,
    'save_checkpoint': True,
    'bias_correction': False,
    'dali': False
}
