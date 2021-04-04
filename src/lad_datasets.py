import pandas as pd
from os.path import join
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from PIL import Image


def get_label_list(label_list_path):
    label_list = pd.read_csv(label_list_path, delimiter=', ', engine='python',
                             names=['label_id', 'label', 'label_ch'])

    label_list['label_code'] = label_list.index

    del label_list['label_ch']

    return label_list


def get_image_list(image_list_path):
    images = pd.read_csv(image_list_path, delimiter=', ', engine='python',
                         names=['image_id', 'label_id', 'y1', 'x1', 'y2', 'x2', 'path'])

    images['bb'] = images['y1'].map(str) + ', ' + \
                   images['x1'].map(str) + ', ' + \
                   images['y2'].map(str) + ', ' + \
                   images['x2'].map(str)

    images['bb'] = images['bb'].map(eval)

    images.drop(columns=['x1', 'y1', 'x2', 'y2'], inplace=True)

    return images


def get_image_attributes(attributes_path):
    img_attributes = pd.read_csv(attributes_path, delimiter=', ', engine='python',
                                 names=['label_id', 'path', 'attributes'])

    img_attributes['attributes'] = img_attributes['attributes']\
        .map(lambda atts: [int(a) for a in atts.strip('[]').split()])

    return img_attributes


def get_image_labels(annotations_dir):
    images = get_image_list(join(annotations_dir, 'images.txt'))
    label_map = get_label_list(join(annotations_dir, 'label_list.txt'))

    image_labels = pd.merge(images, label_map, on='label_id')

    return image_labels


def get_image_attributes_with_bb_coordinates(annotations_dir):
    img_attributes = get_image_attributes(join(annotations_dir, 'attributes.txt'))
    image_list = get_image_list(join(annotations_dir, 'images.txt'))

    del image_list['label_id']

    attributes_bb_coordinates = pd.merge(img_attributes, image_list, on='path')

    return attributes_bb_coordinates


class LadLabelsDataset(Dataset):
    def __init__(self, ds_path, transform=None, crop_bb=True):
        self.image_labels = get_image_labels(join(ds_path, 'LAD_annotations'))
        self.img_dir_path = join(ds_path, 'LAD_images')
        self.transform = transform
        self.crop_bb = crop_bb

    def __getitem__(self, index) -> T_co:
        entry = self.image_labels.iloc[index]
        img_path = join(self.img_dir_path, entry['path'])
        label_code = entry['label_code']

        x = Image.open(img_path).convert('RGB')

        if self.crop_bb:
            y1, x1, y2, x2 = entry['bb']
            x = x.crop((x1, y1, x2, y2))

        if self.transform:
            x = self.transform(x)

        return x, label_code

    def __len__(self):
        return self.image_labels.shape[0]


class LadAttributesDataset(Dataset):
    def __init__(self, ds_path, transform=None, crop_bb=True):
        self.img_attributes = get_image_attributes_with_bb_coordinates(join(ds_path, 'LAD_annotations'))
        self.img_dir_path = join(ds_path, 'LAD_images')
        self.transform = transform
        self.crop_bb = crop_bb

    def __getitem__(self, item):
        entry = self.img_attributes.iloc[item]
        img_path = join(self.img_dir_path, entry['path'])
        attributes = entry['attributes']

        x = Image.open(img_path).convert('RGB')

        if self.crop_bb:
            y1, x1, y2, x2 = entry['bb']
            x = x.crop((x1, y1, x2, y2))

        if self.transform:
            x = self.transform(x)

        return x, torch.tensor(attributes, dtype=torch.float)

    def __len__(self):
        return self.img_attributes.shape[0]
