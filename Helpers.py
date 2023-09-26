import torch
from torch.utils.data import Dataset
import torchvision.datasets as Datasets

import os


def get_datasets(dataset_root, split, train_transform, test_transform):
    data_folders = os.listdir(dataset_root)

    if "train" in data_folders:
        train_path = os.path.join(dataset_root, "train")
        train_set = Datasets.ImageFolder(root=train_path, transform=train_transform)

    else:
        train_set = None
        ValueError("train folder does not exist!")

    if "test" in data_folders:
        test_path = os.path.join(dataset_root, "test")
        test_set = Datasets.ImageFolder(root=test_path, transform=test_transform)
    else:
        test_set = None
        ValueError("test folder does not exist!")

    if "valid" in data_folders:
        val_path = os.path.join(dataset_root, "valid")
        val_set = Datasets.ImageFolder(root=val_path, transform=train_transform)
    else:
        print("-Creating Validation Split!")
        n_val_examples = int(len(train_set) * split)
        n_train_examples = len(train_set) - n_val_examples

        train_set, val_set = torch.utils.data.random_split(train_set, [n_train_examples, n_val_examples],
                                                           generator=torch.Generator().manual_seed(42))

    return train_set, test_set, val_set, len(os.listdir(train_path))
