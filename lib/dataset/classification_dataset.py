# pylint: disable=E1101,R,C
"""
Pytorch customized dataset_name
"""
import glob
import os
import sys
import numpy as np
import torch.utils.data
from random import randint

IMPLEMENTED_DATASET = ["modelnet10", "modelnet40", "shrec15_0.2","shrec15_0.3","shrec15_0.4","shrec15_0.5","shrec15_0.6","shrec15_0.7", "shrec17"]


class ClassificationDataset(torch.utils.data.Dataset):
    """Pytorch customized spherical dataset
    Usage:
    1. Used as single input dataset
        dataset = ClassificationDataset("~/caps3d/data","modelnet10",
            train=True,type="rotate",b=32,target_transform=target_transform)

    2. Used as multi input dataset
        dataset = ClassificationDataset("~/caps3d/data","modelnet10",
            train=True,type="rotate",b=[32,16,8],target_transform=target_transform)

    3. Pick randomly when use rotation repeat dataset
        dataset = ClassificationDataset("~/caps3d/data","modelnet10",
            train=True,type="rotate",pick_randomly=True, b=32,target_transform=target_transform)
    """

    def __init__(
        self,
        root,
        dataset_name,
        train=True,
        type="original",
        b=32,
        target_transform=None,
        ftype="npy",
        **kwargs,
    ):
        if not dataset_name in IMPLEMENTED_DATASET:
            raise ValueError(f"Invalid dataset_name {IMPLEMENTED_DATASET}")
        self.root = os.path.expanduser(root)
        self.target_transform = target_transform
        self.dataset = dataset_name
        self.ftype = ftype
        self.type = type
        partition = "train" if train is True or train == "train" else "test"
        self.pick_randomly = (
            False if "pick_randomly" not in kwargs else kwargs["pick_randomly"]
        )
        # such as prefix/modelnet10/modelnet10_train/rotate
        self.dir = os.path.join(
            self.root, dataset_name, dataset_name + "_" + partition, type
        )
        self.bandwidth = b
        if not self._check_exists():
            raise Exception("Dataset not found for {}.".format(self.dataset))

        # Note: bandwidth can be int or list both, and both work
        self.files = self._search_files()
        self.labels = {}
        for fpath in self.files:
            if isinstance(fpath, list):
                fpath = fpath[0]
            name = fname = os.path.splitext(os.path.basename(fpath))[0]
            if self.ftype == "off":
                # such as:b1_r1_airplane_0001 --> airplane_0001
                name = "_".join(fname.split("_")[2:])
            self.labels[fname] = "_".join(name.split("_")[:-1])  # extract label.

    def _search_files(self):
        """

        Returns:
            list or list(list)
        """
        if isinstance(self.bandwidth, int):
            b = self.bandwidth
        elif isinstance(self.bandwidth, list):
            b = self.bandwidth[0]
        else:
            raise Exception("Wrong bandwidth type")

        files = sorted(
            glob.glob(os.path.join(self.dir, "*.{}".format(self.ftype)))
        )
        if isinstance(self.bandwidth, int):
            # pick randomly from repeated dataset
            if self.pick_randomly and self.type == "rotate":
                file_set = []
                for f in files:
                    head, tail = os.path.split(f)
                    tail = tail.split("_")
                    tail.pop(1)
                    tail = "_".join(tail)
                    file_set.append(os.path.join(head, tail))
                file_set = list(set(file_set))
                random_files = []

                for f in file_set:
                    head, tail = os.path.split(f)
                    tail = tail.split("_")
                    tail.insert(1, "r{}".format(randint(0, 3)))
                    tail = "_".join(tail)
                    random_files.append(os.path.join(head, tail))
                return random_files
            return files

        elif isinstance(self.bandwidth, list):
            multi_input_files = []
            for f in files:
                head, tail = os.path.split(f)
                tail = "b{}_" + "_".join(tail.split("_")[1:])
                multi_input_files.append(
                    [os.path.join(head, tail.format(i)) for i in self.bandwidth]
                )
            return multi_input_files

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            list of img if have multiple transforms
            img else
            target
        """

        choice = self.files[index]
        if isinstance(choice, list):
            img = [np.load(f, allow_pickle=True) for f in choice]
            choice = choice[0]
        elif isinstance(choice, str):
            img = np.load(choice, allow_pickle=True)
        else:
            raise Exception("wrong type for {}th elem: {}".format(index, choice))

        i = os.path.splitext(os.path.basename(choice))[0]
        target = self.labels[i]

        if self.target_transform is not None:
            # transform labels to digit, for example airplane to 0
            # may be a little complicated, this func is initialized from makeDataset
            target = self.target_transform(target)
        # (6, 64, 64)
        return img, target

    def __len__(self):
        return len(self.files)

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir, "*." + self.ftype))
        return len(files) > 0

# if __name__ == '__main__':
