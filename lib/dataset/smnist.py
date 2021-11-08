import gzip
import pickle
import torch
import numpy as np
import os
from torch.utils.data import Dataset


class SMNIST_Dataset(Dataset):
    def __init__(
        self, no_rotate_train=False, no_rotate_test=False, train=True, overlap=False
    ):
        # check path and choice parameters
        if no_rotate_train and no_rotate_test:
            path = "data/s2_mnist_nr_nr.gz"
        elif not no_rotate_train and not no_rotate_test:
            path = "data/s2_mnist_r_r.gz"
        elif no_rotate_train and not no_rotate_test:
            path = "data/s2_mnist_nr_r.gz"
        if overlap:
            path ="data/s2_mnist_overlap_test.gz"
        if not os.path.exists(path):
            raise FileNotFoundError(
                "please put spherical mnist into data/s2_mnist.gz, and run in the project root path!"
            )

        choice = "train" if train else "test"
        if not train and overlap:
            image_key, label_key = "overlap_data", "overlap_labels"
        else:
            image_key, label_key = "images", "labels"

        with gzip.open(path, "rb") as f:
            dataset = pickle.load(f)
            self.data = torch.from_numpy(
                dataset[choice][image_key][:, None, :, :].astype(np.float32)
            )
            self.labels = torch.from_numpy(dataset[choice][label_key].astype(np.int64))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]
