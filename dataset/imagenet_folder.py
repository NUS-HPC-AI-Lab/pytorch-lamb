import copy
import os
import warnings

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.datasets.folder import default_loader


__all__ = ['ImageNetFolder']

# filter warnings for corrupted data
warnings.filterwarnings('ignore')


class ImageNetFolder(dict):
    def __init__(self, root, extra_train_transforms=None):
        train_transforms_pre = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ]
        train_transforms_post = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]
        if extra_train_transforms is not None:
            if not isinstance(extra_train_transforms, list):
                extra_train_transforms = [extra_train_transforms]
            for ett in extra_train_transforms:
                if isinstance(ett, (transforms.LinearTransformation, transforms.Normalize, transforms.RandomErasing)):
                    train_transforms_post.append(ett)
                else:
                    train_transforms_pre.append(ett)
        train_transforms = transforms.Compose(
            train_transforms_pre + train_transforms_post)

        test_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]
        test_transforms = transforms.Compose(test_transforms)

        train = datasets.ImageFolder(
            root=os.path.join(root, 'train'), transform=train_transforms)
        test = datasets.ImageFolder(
            root=os.path.join(root, 'val'), transform=test_transforms)

        super().__init__(train=train, test=test)
