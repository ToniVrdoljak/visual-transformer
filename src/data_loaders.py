import os
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import transforms

from lad_datasets import LadLabelsDataset, LadAttributesDataset, get_image_labels


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
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8, train_split=0.8,
                 sample=False):
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

        lad_dataset = LadLabelsDataset(data_dir, transform=transform, crop_bb=False)

        size = len(lad_dataset)
        train_size = int(size * train_split)
        val_size = size - train_size

        train_ds, val_ds = \
            random_split(lad_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        if split == 'train' and sample:
            print('Creating weighted sampler')
            image_labels = get_image_labels(os.path.join(data_dir, 'LAD_annotations'))
            inverse_weights = torch.tensor(list(image_labels.label_code.value_counts().sort_index()), dtype=torch.float)
            weights = 1 / inverse_weights

            classes = list(image_labels['label_code'])

            tc, _ = random_split(classes, [train_size, val_size], generator=torch.Generator().manual_seed(42))

            sampler = WeightedRandomSampler(weights=weights[tc], num_samples=train_size, replacement=True,
                                            generator=torch.Generator().manual_seed(42))

        super(LadLabelsDataLoader, self).__init__(
            dataset=train_ds if split == 'train' else val_ds,
            batch_size=batch_size,
            shuffle=True if split == 'train' and not sample else False,
            sampler=sampler if split == 'train' and sample else None,
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

        lad_dataset = LadAttributesDataset(data_dir, transform=transform, crop_bb=False)

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
