import os
import pickle as pkl

import numpy as np

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

try:
    from itertools import zip_longest as zip_longest
except:
    from itertools import izip_longest as zip_longest


def load_mean_std(dataset_name, train, bw=32):
    root = "/home/qiangzibro/caps3d/data/"
    _type = "train" if train is True or train == "train" else "test"
    file = os.path.join(root, "info.pkl")
    info = _load_info(file)
    try:
        mean_std = info[dataset_name][bw][_type]
        mean, std = mean_std["mean"], mean_std["std"]
        return mean, std
    except:
        raise Exception("Please compute mean and std first")


def _load_info(file):
    try:
        info = pkl.load(open(file, "rb"))
    except:
        print("file non-existent")
        info = {}
    return info


def compute_mean_std(dataset, max_workers=None):
    N = len(dataset)

    def _compute_mean(i):
        # print(f"compute mean for {i}")
        data, _ = dataset[i]
        return np.mean(data, axis=(1, 2))

    def _compute_std(i):
        # print(f"compute std for {i}")
        data, _ = dataset[i]
        return ((data - mean) ** 2).mean(axis=(1, 2))

    if max_workers:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            result = executor.map(_compute_mean, range(N))
    else:
        result = [_compute_mean(i) for i in range(N)]

    mean = sum(result) / N
    mean = mean.reshape(-1, 1, 1)

    if max_workers:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            result = executor.map(_compute_std, range(N))
    else:
        result = [_compute_std(i) for i in range(N)]
    std = sum(result) / N
    std = np.sqrt(std)
    std = std.reshape(-1, 1, 1)

    return mean, std


def compute_mean_and_std_for(dataset_name, train, bws, save=True, **kargs):
    """

    Args:
        dataset_name:
        train:
        bws:
        save:
        **kargs: type

    Returns:

    """
    from lib.dataset import makeDataset

    for b in bws:
        dataset = makeDataset(
            train=train, b=b, dataset_name=dataset_name, normalize=False, **kargs
        )
        mean, std = compute_mean_std(dataset=dataset, max_workers=None)
        print("----------------------------------------------")
        print(
            "dataset_name:{}, {}, {}, {}".format(
                dataset_name, kargs["type"], "train" if train else "test", b
            )
        )
        print("mean:", mean.squeeze())
        print("std:", std.squeeze())
        if save:
            _save_mean_std(mean, std, b, dataset_name, kargs["type"], train=train)


def _save_mean_std(
    mean,
    std,
    bw,
    dataset_name,
    type,
    root="/home/qiangzibro/caps3d/data/",
    train=True,
):
    """
    the mean and std is stored as:
    {
        "modelnet10":{
            "rotate":{
                32:{
                    "train":{"mean":mean, "std":std},
                    "test":{"mean":mean, "std":std}
                },
                16:{...}
            }
        }
    }
    so you may get the mean and std by:
    mean_std = info[dataset_name][partition][type][bandwidths]
    mean, std = mean_std["mean"], mean_std["std"]
    """
    partition = "train" if train else "test"
    file = os.path.join(root, "info.pkl")
    info = _load_info(file)

    if dataset_name not in info.keys():
        info[dataset_name] = {}
    if partition not in info[dataset_name].keys():
        info[dataset_name][partition] = {}
    specific_dataset = info[dataset_name][partition]
    if type in specific_dataset.keys():
        specific_dataset[type][bw] = {"mean": mean, "std": std}
    else:
        specific_dataset[type] = {bw: {"mean": mean, "std": std}}
    with open(file, "wb") as f:
        pkl.dump(info, f)
