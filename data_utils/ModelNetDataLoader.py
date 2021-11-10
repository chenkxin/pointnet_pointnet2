'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
# import argparse
warnings.filterwarnings('ignore')
import glob
import os
import trimesh
import sys
import torch.utils.data
from random import randint
# def parse_args():
#     '''PARAMETERS'''
#     parser = argparse.ArgumentParser('training')
#     parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
#     parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
#     parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
#     parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
#     parser.add_argument(
#         "--dataset_name",
#         type=str,
#         default="modelnet10",
#         choices=["modelnet10", "modelnet40", "shrec15_0.2", "shrec15_0.3", "shrec15_0.4", "shrec15_0.5", "shrec15_0.6",
#                  "shrec15_0.7"],
#     )
#     # parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
#     parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
#     parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
#     parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
#     parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
#     parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
#     parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
#     parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
#     parser.add_argument('--random_rotations', action='store_true', default=True, help='save data offline')
#     parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
#     return parser.parse_args()
IMPLEMENTED_DATASET = ["modelnet10", "modelnet40", "shrec15_0.2", "shrec15_0.3", "shrec15_0.4", "shrec15_0.5", "shrec15_0.6", "shrec15_0.7"]
modelnet10_classes = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]
modelnet40_classes = [
    "tv_stand",
    "guitar",
    "lamp",
    "cup",
    "bed",
    "desk",
    "vase",
    "bottle",
    "bookshelf",
    "chair",
    "tent",
    "sink",
    "curtain",
    "wardrobe",
    "glass_box",
    "door",
    "range_hood",
    "mantel",
    "dresser",
    "plant",
    "stairs",
    "bench",
    "bowl",
    "night_stand",
    "table",
    "flower_pot",
    "airplane",
    "cone",
    "xbox",
    "radio",
    "laptop",
    "bathtub",
    "monitor",
    "person",
    "toilet",
    "car",
    "stool",
    "keyboard",
    "piano",
    "sofa",
]
shrec15_classes = [
    "sumotori",
    "paper",
    "bull",
    "mouse",
    "horse",
    "man",
    "aligator",
    "nunchaku",
    "robot",
    "santa",
    "dinosaur",
    "hand",
    "armadillo",
    "spider",
    "frog",
    "alien",
    "tortoise",
    "twoballs",
    "ants",
    "dragon",
    "mantaray",
    "elephant",
    "lamp",
    "ring",
    "watch",
    "woman",
    "octopus",
    "weedle",
    "centaur",
    "dinoske",
    "snake",
    "woodman",
    "deer",
    "glasses",
    "mermaid",
    "pliers",
    "kangaroo",
    "bird",
    "dog",
    "giraffe",
    "rabbit",
    "chick",
    "camel",
    "shark",
    "gorilla",
    "cat",
    "flamingo",
    "scissor",
]

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def _get_classes(dataset_name):
    if dataset_name == "modelnet10":
        classes = modelnet10_classes
    elif dataset_name == "modelnet40":
        classes = modelnet40_classes
    elif dataset_name == "shrec15_0.2":
        classes = shrec15_classes
    elif dataset_name == "shrec15_0.3":
        classes = shrec15_classes
    elif dataset_name == "shrec15_0.4":
        classes = shrec15_classes
    elif dataset_name == "shrec15_0.5":
        classes = shrec15_classes
    elif dataset_name == "shrec15_0.6":
        classes = shrec15_classes
    elif dataset_name == "shrec15_0.7":
        classes = shrec15_classes
    else:
        raise ValueError(f"No such dataset {dataset_name}")
    return classes

class target_transform():
    def __init__(self,dataset_name):
        self.dataset_name = dataset_name
    def __getitem__(self, index):
        classes = _get_classes(self.dataset_name)
        return classes.index(index)

def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate

    :param a, b, c: ZYZ-Euler angles
    """

    def z(a):
        return np.array(
            [
                [np.cos(a), np.sin(a), 0, 0],
                [-np.sin(a), np.cos(a), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def y(a):
        return np.array(
            [
                [np.cos(a), 0, np.sin(a), 0],
                [0, 1, 0, 0],
                [-np.sin(a), 0, np.cos(a), 0],
                [0, 0, 0, 1],
            ]
        )

    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
    if hom_coord:
        return r
    else:
        return r[:3, :3]

def rnd_rot():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, np.arccos(z), c, True)
    return rot

class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train'):
        if not args.dataset_name in IMPLEMENTED_DATASET:
            raise ValueError(f"Invalid dataset_name {IMPLEMENTED_DATASET}")
        self.root = os.path.expanduser(root)
        self.target_transform = target_transform(dataset_name=args.dataset_name)
        self.npoints = args.num_point
        self.random_rotations = args.random_rotations
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.dataset_name = args.dataset_name
        self.split = split
        #self.ftype = ftype
        # such as prefix/modelnet10/modelnet10_train/
        self.dir = os.path.join(
            self.root, self.dataset_name, self.dataset_name + "_" + split
        )
        if not self._check_exists():
            raise Exception("Dataset not found for {}.".format(self.dataset_name))
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.off")))
        self.labels = {}
        for fpath in self.files:
            if isinstance(fpath, list):
                fpath = fpath[0]
            name = fname = os.path.splitext(os.path.basename(fpath))[0]
            self.labels[fname] = "_".join(name.split("_")[:-1])  # extract label.

    def __len__(self):
        return len(self.files)

    def _get_item(self, index):
        """
           Args:
               index
           Returns:
            point_set, label
        """
        choice = self.files[index]
        if isinstance(choice, str):
            mesh = trimesh.load_mesh(choice)
            point_set = np.array(mesh.vertices.astype(np.float32))
        else:
            raise Exception("wrong type for {}th elem: {}".format(index, choice))
        if self.random_rotations and self.split =='test':
            mesh.apply_transform(rnd_rot())
            point_set = np.array(mesh.vertices.astype(np.float32))

        i = os.path.splitext(os.path.basename(choice))[0]
        target = self.labels[i]

        if self.target_transform is not None:
            # transform labels to digit, for example airplane to 0
            # may be a little complicated, this func is initialized from makeDataset
            target = self.target_transform.__getitem__(target)

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]

        point_set= pc_normalize(point_set)
        if not self.use_normals:
            point_set = point_set

        return point_set, target

    def __getitem__(self, index):
        return self._get_item(index)

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir, "*.off"))
        return len(files) > 0

# if __name__ == '__main__':
#     args = parse_args()
#     m = ModelNetDataLoader(root='~/Pointnet_Pointnet2_pytorch/data',args = args)
#     a = m.__getitem__(0)
#     print(m)