# PyTorch LAMB implementation for ImageNet/ResNet-50

This is an implementation of [LAMB](https://arxiv.org/pdf/1904.00962.pdf) optimizer by PyTorch for ImageNet/ResNet-50 training.

## Training
We use horovod to run distributed training:

```
horovodrun -np N python train.py --dataset_path=your_imagenet_folder
```

You can refer the paper for some important settings of hyper-parameters, like learning rate and warmup epochs.

You can also simply use LAMB in your own projects, just use optim.lamb.create_lamb_optimizer to create LAMB optimizer. We implement the feature of excluding some layers (like BatchNorm, LayerNorm and bias layers) from weight decay, which influences the training.

