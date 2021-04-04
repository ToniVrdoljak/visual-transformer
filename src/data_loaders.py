import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import transforms

from lad_datasets import LadLabelsDataset, LadAttributesDataset


__all__ = ['CIFAR10DataLoader', 'ImageNetDataLoader', 'CIFAR100DataLoader',
           'LadLabelsDataLoader', 'LadAttributesDataLoader']


class CIFAR10DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = CIFAR10(root=data_dir, train=train, transform=transform, download=True)

        super(CIFAR10DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers)


class CIFAR100DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = CIFAR100(root=data_dir, train=train, transform=transform, download=True)

        super(CIFAR100DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers)


class ImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):

        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = ImageFolder(root=os.path.join(data_dir, split), transform=transform)
        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers)


class LadLabelsDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8, train_split=0.8):
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        lad_dataset = LadLabelsDataset(data_dir, transform=transform)

        size = len(lad_dataset)
        train_size = int(size * train_split)
        val_size = size - train_size

        train_ds, val_ds = \
            random_split(lad_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        super(LadLabelsDataLoader, self).__init__(
            dataset=train_ds if split == 'train' else val_ds,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers)


class LadAttributesDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8, train_split=0.8):
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        lad_dataset = LadAttributesDataset(data_dir, transform=transform)

        size = len(lad_dataset)
        train_size = int(size * train_split)
        val_size = size - train_size

        train_ds, val_ds = \
            random_split(lad_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        super(LadAttributesDataLoader, self).__init__(
            dataset=train_ds if split == 'train' else val_ds,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers)


if __name__ == '__main__':
    data_loader = ImageNetDataLoader(
        data_dir='/home/hchen/Projects/vat_contrast/data/ImageNet',
        split='val',
        image_size=384,
        batch_size=16,
        num_workers=0)

    for images, targets in data_loader:
        print(targets)
