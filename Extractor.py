from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import os

DATA_PATH = f'{os.getcwd()}/Data'

TEST_PATH = f'{DATA_PATH}/test'
TRAIN_PATH = f'{DATA_PATH}/train'
VALID_PATH = f'{DATA_PATH}/valid'


def get_train_images():
    train_dataset = ImageFolder(root = TRAIN_PATH, transform = T.ToTensor())
    train_random_sampler = RandomSampler(train_dataset)
    return DataLoader(
        dataset=train_dataset,
        batch_size=16,
        sampler=train_random_sampler,
        num_workers=4,
    )
# test = get_test_images()
# train = get_train_images()
# valid = get_valid_images()


get_train_images()
