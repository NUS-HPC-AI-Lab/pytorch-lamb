import argparse
import math
import os
import random
import shutil
from datetime import datetime
from config import configs

import numpy as np
import horovod.torch as hvd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torchvision import models
from tqdm import tqdm

from dataset import ImageNetFolder, make_meters, DaliImageNet
from optim.lamb import create_lamb_optimizer
from optim import lr_scheduler
from loss import LabelSmoothLoss

METRIC = 'acc/test_top1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--total_batch_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--dataset_path')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--num_threads', type=int)
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--lr_scaling')
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--warmup_epochs', type=float)
    parser.add_argument('--bias_correction',
                        default=None, action='store_true')
    parser.add_argument('--save_checkpoint',
                        default=None, action='store_true')
    parser.add_argument('--dali',
                        default=None, action='store_true')
    args = parser.parse_args()

    ##################
    # Update configs #
    ##################
    for k, v in configs.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    for k, v in vars(args).items():
        printr(f'[{k}] = {v}')

    if args.device is not None and args.device != 'cpu':
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        cudnn.benchmark = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        cudnn.deterministic = True
        cudnn.benchmark = False

    num_batches_per_step = args.total_batch_size // (
        args.batch_size * hvd.size())
    if num_batches_per_step * args.batch_size * hvd.size() != args.total_batch_size:
        raise ValueError(
            f'total_batch_size({args.total_batch_size}) is not integer multiples of batch_size({args.batch_size}) * GPUs({hvd.size()})')

    save_path = f'runs/lamb-{args.total_batch_size}{args.suffix}.np{hvd.size()}'
    printr(f'[save_path] = {save_path}')
    checkpoint_path = os.path.join(save_path, 'checkpoints')
    checkpoint_path_fmt = os.path.join(
        checkpoint_path, f'e{"{epoch}"}-r{hvd.rank()}.pth'
    )
    latest_pth_path = os.path.join(
        checkpoint_path, f'latest-r{hvd.rank()}.pth'
    )
    best_pth_path = os.path.join(
        checkpoint_path, f'best-r{hvd.rank()}.pth'
    )
    os.makedirs(checkpoint_path, exist_ok=True)

    if args.evaluate:
        latest_pth_path = best_pth_path

    #####################################################################
    # Initialize DataLoaders, Model, Criterion, LRScheduler & Optimizer #
    #####################################################################

    printr(f'\n==> creating dataset from "{args.dataset_path}"')
    if args.dali:
        dataset = DaliImageNet(args.dataset_path,
                               batch_size=args.batch_size,
                               train_batch_size=args.batch_size * num_batches_per_step,
                               shard_id=hvd.rank(),
                               num_shards=hvd.size(),
                               num_workers=args.num_workers)
    else:
        dataset = ImageNetFolder(args.dataset_path)
        # Horovod: limit # of CPU threads to be used per worker.
        loader_kwargs = {'num_workers': args.num_workers,
                         'pin_memory': True} if args.device == 'cuda' else {}
        # When supported, use 'forkserver' to spawn dataloader workers
        # instead of 'fork' to prevent issues with Infiniband implementations
        # that are not fork-safe
        if (loader_kwargs.get('num_workers', 0) > 0 and
                hasattr(mp, '_supports_context') and
                mp._supports_context and
                'forkserver' in mp.get_all_start_methods()):
            loader_kwargs['multiprocessing_context'] = 'forkserver'
        printr(f'\n==> loading dataset "{loader_kwargs}""')
    torch.set_num_threads(args.num_threads)
    if args.dali:
        samplers, loaders = {split: None for split in dataset}, dataset
    else:
        samplers, loaders = {}, {}
        for split in dataset:
            # Horovod: use DistributedSampler to partition data among workers.
            # Manually specify `num_replicas=hvd.size()` and `rank=hvd.rank()`.
            samplers[split] = torch.utils.data.distributed.DistributedSampler(
                dataset[split], num_replicas=hvd.size(), rank=hvd.rank())
            loaders[split] = torch.utils.data.DataLoader(
                dataset[split], batch_size=args.batch_size * (
                    num_batches_per_step if split == 'train' else 1),
                sampler=samplers[split],
                drop_last=(num_batches_per_step > 1
                           and split == 'train'),
                **loader_kwargs
            )

    printr(f'\n==> creating model "resnet50"')
    model = models.resnet50()
    model = model.to(args.device)

    criterion = LabelSmoothLoss(smoothing=0.1).to(args.device)
    # Horovod: scale learning rate by the number of GPUs.
    lr = args.base_lr
    if args.lr_scaling == 'sqrt':
        lr *= math.sqrt(num_batches_per_step * hvd.size())
    elif args.lr_scaling == 'linear':
        lr *= num_batches_per_step * hvd.size()
    printr(f'\n==> creating optimizer LAMB with LR = {lr}')

    optimizer = create_lamb_optimizer(
        model, lr, weight_decay=args.weight_decay, bias_correction=args.bias_correction)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        backward_passes_per_step=num_batches_per_step,
        op=hvd.Average
    )

    # resume from checkpoint
    last_epoch, best_metric = -1, None
    if os.path.exists(latest_pth_path):
        printr(f'\n[resume_path] = {latest_pth_path}')
        checkpoint = torch.load(latest_pth_path)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        last_epoch = checkpoint.get('epoch', last_epoch)
        best_metric = checkpoint.get('meters', {}).get(
            f'{METRIC}_best', best_metric)
        # Horovod: broadcast parameters.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    else:
        printr('\n==> train from scratch')
        # Horovod: broadcast parameters & optimizer state.
        printr('\n==> broadcasting paramters and optimizer state')
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    num_steps_per_epoch = len(loaders['train'])
    warmup_lr_epochs = getattr(args, 'warmup_epochs', 0)

    last = max((last_epoch - warmup_lr_epochs + 1)
               * num_steps_per_epoch - 2, -1)
    decay_steps = args.num_epochs * num_steps_per_epoch
    warmup_steps = warmup_lr_epochs
    if warmup_lr_epochs > 0:
        warmup_steps *= num_steps_per_epoch

    scheduler = lr_scheduler.PolynomialWarmup(
        optimizer, decay_steps, warmup_steps, end_lr=0.0, power=1.0, last_epoch=last)

    ############
    # Training #
    ############

    training_meters = make_meters()
    meters = evaluate(model, device=args.device, meters=training_meters,
                      loader=loaders['test'], split='test', dali=args.dali)
    for k, meter in meters.items():
        printr(f'[{k}] = {meter:.2f}')
    if args.evaluate or last_epoch >= args.num_epochs:
        return

    if hvd.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        tensorboard_path = os.path.join(save_path, timestamp)
        writer = SummaryWriter(tensorboard_path)
    else:
        writer = None

    for current_epoch in range(last_epoch + 1, args.num_epochs):
        printr(f'\n==> training epoch {current_epoch + 1}/{args.num_epochs}')

        train(model=model, loader=loaders['train'],
              device=args.device, epoch=current_epoch,
              sampler=samplers['train'], criterion=criterion,
              optimizer=optimizer, scheduler=scheduler,
              batch_size=args.batch_size,
              num_batches_per_step=num_batches_per_step,
              num_steps_per_epoch=num_steps_per_epoch,
              warmup_lr_epochs=warmup_lr_epochs,
              schedule_lr_per_epoch=False,
              writer=writer, quiet=hvd.rank() != 0, dali=args.dali)

        meters = dict()
        for split, loader in loaders.items():
            if split != 'train':
                meters.update(evaluate(model, loader=loader,
                                       device=args.device,
                                       meters=training_meters,
                                       split=split, quiet=hvd.rank() != 0, dali=args.dali))

        best = False
        if best_metric is None or best_metric < meters[METRIC]:
            best_metric, best = meters[METRIC], True
        meters[f'{METRIC}_best'] = best_metric

        if writer is not None:
            num_inputs = ((current_epoch + 1) * num_steps_per_epoch
                          * num_batches_per_step
                          * args.batch_size * hvd.size())
            print('')
            for k, meter in meters.items():
                print(f'[{k}] = {meter:.2f}')
                writer.add_scalar(k, meter, num_inputs)

        checkpoint = {
            'epoch': current_epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'meters': meters
        }

        # save checkpoint
        if args.save_checkpoint:
            checkpoint_path = checkpoint_path_fmt.format(epoch=current_epoch)
            torch.save(checkpoint, checkpoint_path)
            shutil.copyfile(checkpoint_path, latest_pth_path)
            if best:
                shutil.copyfile(checkpoint_path, best_pth_path)
            if current_epoch >= 3:
                os.remove(
                    checkpoint_path_fmt.format(epoch=current_epoch - 3)
                )
            printr(f'[save_path] = {checkpoint_path}')


def train(model, loader, device, epoch, sampler, criterion, optimizer,
          scheduler, batch_size, num_batches_per_step, num_steps_per_epoch, warmup_lr_epochs, schedule_lr_per_epoch, writer=None, quiet=True, dali=False):
    step_size = num_batches_per_step * batch_size
    num_inputs = epoch * num_steps_per_epoch * step_size * hvd.size()
    _r_num_batches_per_step = 1.0 / num_batches_per_step

    if sampler:
        sampler.set_epoch(epoch)
    model.train()
    for step, data in enumerate(tqdm(
            loader, desc='train', ncols=0, disable=quiet)):
        if dali:
            inputs, targets = data[0]['data'], data[0]['label']
        else:
            inputs, targets = data
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        loss = torch.tensor([0.0])
        for b in range(0, step_size, batch_size):
            _inputs = inputs[b:b+batch_size]
            _targets = targets[b:b+batch_size]
            _outputs = model(_inputs)
            _loss = criterion(_outputs, _targets)
            _loss.mul_(_r_num_batches_per_step)
            _loss.backward()
            loss += _loss.item()
        optimizer.step()

        # write train loss log
        loss = hvd.allreduce(loss, name='loss').item()
        if writer is not None:
            num_inputs += step_size * hvd.size()
            writer.add_scalar('loss/train', loss, num_inputs)
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr/train', lr, num_inputs)

        adjust_learning_rate(scheduler, epoch=epoch, step=step,
                             schedule_lr_per_epoch=schedule_lr_per_epoch)


def evaluate(model, loader, device, meters, split='test', quiet=True, dali=False):
    _meters = {}
    for k, meter in meters.items():
        meter.reset()
        _meters[k.format(split)] = meter
    meters = _meters

    model.eval()

    with torch.no_grad():
        for data in tqdm(loader, desc=split, ncols=0, disable=quiet):
            if dali:
                inputs, targets = data[0]['data'], data[0]['label']
            else:
                inputs, targets = data
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            for meter in meters.values():
                meter.update(outputs, targets)

    for k, meter in meters.items():
        data = meter.data()
        for dk, d in data.items():
            data[dk] = \
                hvd.allreduce(torch.tensor([d]), name=dk, op=hvd.Sum).item()
        meter.set(data)
        meters[k] = meter.compute()
    return meters


def adjust_learning_rate(scheduler, epoch, step,
                         schedule_lr_per_epoch=False):
    if schedule_lr_per_epoch and (step > 0 or epoch == 0):
        return
    else:
        scheduler.step()


def printr(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs)


if __name__ == '__main__':
    hvd.init()
    main()
