import copy
import os
import warnings

import lmdb
import six
import pyarrow as pa
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


__all__ = ['ImageNetFolder']

# filter warnings for corrupted data
warnings.filterwarnings('ignore')


class ImageFolderLMDB(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.db_path = root
        self.transform = transform
        self.target_transform = target_transform
        self.env = None
        self.setup_attr()

    def setup_attr(self):
        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                        readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))
        env.close()

    def setup_env(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.env is None:
            self.setup_env()
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class ImageNetFolder(dict):
    def __init__(self, root, image_size=224, extra_train_transforms=None, lmdb=False):
        train_transforms_pre = [
            transforms.RandomResizedCrop(image_size),
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
            transforms.Resize(int(image_size / 0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]
        test_transforms = transforms.Compose(test_transforms)

        dataset_cls = ImageFolderLMDB if lmdb else datasets.ImageFolder

        train = dataset_cls(
            root=os.path.join(root, 'train'), transform=train_transforms)
        test = dataset_cls(
            root=os.path.join(root, 'val'), transform=test_transforms)

        super().__init__(train=train, test=test)
