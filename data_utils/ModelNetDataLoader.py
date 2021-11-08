'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

import argparse
warnings.filterwarnings('ignore')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_ssg', help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="modelnet10",
        choices=["modelnet10", "modelnet40", "shrec15_0.2", "shrec15_0.3", "shrec15_0.4", "shrec15_0.5", "shrec15_0.6",
                 "shrec15_0.7"],
    )
    return parser.parse_args()

import glob
import os
import sys
import torch.utils.data
from random import randint

IMPLEMENTED_DATASET = ["modelnet10", "modelnet40", "shrec15_0.2","shrec15_0.3","shrec15_0.4","shrec15_0.5","shrec15_0.6","shrec15_0.7"]

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
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
    xyz = point[:,:3]
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

def target_transform(x,dataset_name):
    classes = _get_classes(dataset_name)
    return classes.index(x)


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = os.path.expanduser(root)
        self.target_transform = target_transform( dataset_name= args.dataset_name)
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.dataset_name =args.dataset_name

        

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            point_set= np.loadtxt('airplane_0001.off', delimiter=' ', skiprows=1, usecols=range(3)).astype(np.float32)
            # point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    args = parse_args()
    # args.log_dir = "pointnet2_cls_ssg"
    data = ModelNetDataLoader('./data/modelnet40_normal_resampled/', split='train', args=args)

    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)

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