# PyTorch LAMB implementation for ImageNet/ResNet-50

This is an implementation of [LAMB](https://arxiv.org/pdf/1904.00962.pdf) optimizer by PyTorch for ImageNet/ResNet-50 training.

## Training
We use horovod to run distributed training:

```
horovodrun -np N python train.py
```

We use config file (`config.py`) and command line arguments to set parameters. `config.py` defines the default values and command line arguments will overwrite the corresponding values. For example, if we pass batch_size=32 to arguments, then batch_size set in `config.py` will be ignored.

You can refer the paper for some important settings of hyper-parameters, like learning rate and warmup epochs.

You can also simply use LAMB in your own projects, just use optim.lamb.create_lamb_optimizer to create LAMB optimizer. We implement the feature of excluding some layers (like BatchNorm, LayerNorm and bias layers) from weight decay, which influences the training.

## Results

| batch size | LR              | warmup epochs | top1 acc                 | top5 acc |
| ---------- | --------------- | ------------- | ------------------------ | -------- |
| 4K         | 4/(2^1.5 * 100) | 2.5           | 77.14                    | 93.50    |
| 8K         | 0.02            | 5             | 77.31                    | 93.64    |
| 16K        | 4/(2^0.5 * 100) | 10            | 76.91                    | 93.30    |
| 32K        | 0.04            | 20            | 76.71                    | 93.00    |
