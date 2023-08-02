import torch.utils.data as data
from torchvision import transforms

from PIL import Image
import glob
import random
import os

random.seed(1143)


def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*")

    train_list = image_list_lowlight

    random.shuffle(train_list)

    return train_list


class lowlight_loader(data.Dataset):
    def __init__(self, lowlight_images_path, patch_size):

        self.train_list = populate_train_list(lowlight_images_path)
        self.patch = patch_size

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        # fetch image
        fn = self.data_list[index]
        im = Image.open(fn).convert("RGB")
        transformer = transforms.Compose(
            [
                transforms.Resize(size=(self.patch, self.patch)),
                transforms.ToTensor(),
            ]
        )
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.data_list)


def populate_train_list_contrast(lowlight_images_path):
    image_list_lowlight_low = glob.glob(os.path.join(lowlight_images_path, 'low', "*"))
    image_list_lowlight_normal = glob.glob(os.path.join(lowlight_images_path, 'normal', "*"))

    train_list = image_list_lowlight_low + image_list_lowlight_normal

    random.shuffle(train_list)

    return train_list


class lowlight_loader_contrast(data.Dataset):
    def __init__(self, lowlight_images_path, patch_size):

        self.train_list = populate_train_list_contrast(lowlight_images_path)
        self.patch = patch_size

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        # fetch image
        fn = self.data_list[index]
        im = Image.open(fn).convert("RGB")
        transformer = transforms.Compose(
            [
                transforms.Resize(size=(self.patch, self.patch)),
                transforms.ToTensor(),
            ]
        )
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.data_list)
