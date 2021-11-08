import glob
import os
import random
import sys
import time
from random import choice, choices
from torchvision import datasets
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy import misc

from lib.dataset.transforms import ProjectOnSphere, ToMesh, make_sgrid


def set_color(mesh, index_tri):
    # unmerge so viewer doesn't smooth
    mesh.unmerge_vertices()
    # make mesh white- ish
    mesh.visual.face_colors = [255, 255, 255, 255]
    mesh.visual.face_colors[index_tri] = [0, 255, 0, 255]


def make_insects(mesh, sgrid, index, b=32):
    # 光线投射
    ray_origins, ray_directions = sgrid[index], -sgrid[index]
    index_tri, index_ray, locations = mesh.ray.intersects_id(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False,
        return_locations=True,
    )

    #     print('The rays with index: {} hit the triangles stored at mesh.faces[{}]'.format(index_ray, index_tri))
    #     print('总共有{}条光线打在了物体上(总共:{}条光线)'.format(len(index_ray), len(sgrid)))
    return ray_origins, ray_directions, index_tri, index_ray, locations


def make_scene(mesh, sgrid, index, b=32):
    ray_origins, ray_directions, index_tri, index_ray, locations = make_insects(
        mesh, sgrid, index, b
    )

    # 组合成一个场景
    ray_visualize = trimesh.load_path(
        np.hstack((ray_origins, [[0, 0, 0] for i in range(len(index))])).reshape(
            -1, 2, 3
        )
    )
    set_color(mesh, index_tri)
    scene = trimesh.Scene([mesh, trimesh.load_path(sgrid), ray_visualize])
    return scene


def check(mesh):
    k = 1
    b = 32
    sgrid = make_sgrid(b=b, alpha=0, beta=0, gamma=0)
    index = random.choices(range(len(sgrid)), k=len(sgrid) if k is None else k)
    return make_scene(mesh, sgrid, index, b=b)


def with_preprocess(path, b=32, k=None):
    #     print("-----------------")
    #     print("用ToMesh处理面片")
    #     print("-----------------")
    sgrid = make_sgrid(b=b, alpha=0, beta=0, gamma=0)

    mesh = ToMesh(True, 0)(path)
    if k:
        index = random.choices(range(len(sgrid)), k=k)  # 随机选取k个点进行可视化
        return make_scene(mesh, sgrid, index, b=b)
    else:
        return mesh


def without_preprocess(path, b=32, k=None):
    #     print("-----------------")
    #     print("不用ToMesh处理面片")
    #     print("-----------------")
    sgrid = make_sgrid(b=b, alpha=0, beta=0, gamma=0)
    index = random.choices(
        range(len(sgrid)), k=len(sgrid) if k is None else k
    )  # 随机选取k个点进行可视化
    mesh = trimesh.load_mesh(path)
    return make_scene(mesh, sgrid, index, b=b)


def get_image_from_mesh(mesh, unit_sphere=False):
    if unit_sphere:
        sgrid = make_sgrid(b=32, alpha=0, beta=0, gamma=0)
        #scene = make_scene_without_lines(mesh, sgrid)
    else:
        scene = trimesh.Scene([mesh])
    data = scene.save_image(visible=True)
    rendered = Image.open(trimesh.util.wrap_as_stream(data)).convert("RGB")
    return rendered


def save_image(mesh, filename):
    get_image_from_mesh(mesh).save(filename)


def imgs_to_row(ims):
    """把一个PIL列表图片汇总成一行图片"""
    widths, heights = zip(*(i.size for i in ims))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in ims:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def imgs_to_col(ims):
    """把一个PIL列表图片汇总成一列图片"""
    widths, heights = zip(*(i.size for i in ims))

    total_width = max(widths)
    max_height = sum(heights)

    new_im = Image.new("RGB", (total_width, max_height))
    y_offset = 0
    for im in ims:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


def find_classes_in_dataset(
    root="/home/qiangzibro/caps3d/data/modelnet40/modelnet40_train/",
):
    files = sorted(glob.glob(os.path.join(root, "*.off")))
    labels = {}
    c = set()
    for fpath in files:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        c_ = "_".join(fname.split("_")[:-1])  # extract label.
        c.add(c_)
        labels[fname] = c_
    return list(c)


def find_classes(name="modelnet40", train=True):
    if train:
        return find_classes_in_dataset(
            root=f"/home/qiangzibro/caps3d/data/{name}/{name}_train/"
        )
    else:
        return find_classes_in_dataset(
            root=f"/home/qiangzibro/caps3d/data/{name}/{name}_test/"
        )


def save_images_in(dataset, k, train=True):
    paths = [
        choice(glob.glob(f"data/{dataset}/{dataset}_train/{cls}_*.off"))
        for cls in find_classes(dataset, train)
    ]
    random.shuffle(paths)

    paths = paths[:k]
    save_images(paths, dataset)


def save_images(paths, filename="save"):
    IMAGES = []
    for i, mesh_object_path in enumerate(paths):
        images = []
        name = mesh_object_path.split("/")[-1].split("_")[0]
        name = "logs/pics/" + name

        mesh = trimesh.load_mesh(mesh_object_path)
        images.append(get_image_from_mesh(mesh))

        mesh = ToMesh(True, 0)(mesh_object_path)
        images.append(get_image_from_mesh(mesh))

        mesh.apply_translation(-mesh.centroid)
        images.append(get_image_from_mesh(mesh))

        IMAGES.append(images)

    tmp = []
    for images in IMAGES:
        tmp.append(imgs_to_row(images))
    fig = imgs_to_col(tmp)
    fig.save(f"logs/pics/{filename}.jpg")


def plot_smnist(train=False, overlap=True):
    """
    A script for viewing spherical images of mnist
    """
    import pickle, gzip, matplotlib.pyplot as plt
    from mayavi import mlab
    import numpy as np
    import math

    def create_shpere(b=60):
        # Make sphere, choose colors
        phi, theta = np.mgrid[0 : np.pi : b * 1j, 0 : 2 * np.pi : b * 1j]
        x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
        return x, y, z

    choice = "train" if train else "test"
    if not train and overlap:
        image_key, label_key = "overlap_data", "overlap_labels"
    else:
        image_key, label_key = "images", "labels"

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1000, 800))
    index = 3
    x, y, z = create_shpere()
    with gzip.open("/home/chenhao/caps3d/data/s2_mnist_nr_nr.gz", "rb") as f:
        dataset = pickle.load(f)
        # im = dataset['train']['images'][index]
        # mlab.mesh(x, y, z, scalars=im, colormap="coolwarm")
        # print(f"This image is {label.item()}")

        #for n in range(10):
        n =76
        print(n)
        im = dataset[choice][image_key][n]
        label = dataset[choice][label_key][n]
        im = np.array(im, dtype=np.dtype("uint8"))
        print(label)
        mlab.mesh(x, y + n * 2 + 0.2, z, scalars=im, colormap="coolwarm")
        #mlab.mesh(x, y, z, scalars=im, colormap="coolwarm")
        mlab.view(0,170,10)
        #filename = "smnist_train_" + "n_" + str(n) + "_lable_" + str(label) + ".png"
        #mlab.savefig(filename)
        mlab.show()
        # directly on 2d
        #plt.imshow(im,cmap ='gray')
        #plt.show()

def plot_mnist(train=False, overlap=True):
    """
    A script for viewing spherical images of mnist
    """
    import pickle, gzip, matplotlib.pyplot as plt
    from mayavi import mlab
    import numpy as np

    choice = "train" if train else "test"
    if not train and overlap:
        image_key, label_key = "overlap_data", "overlap_labels"
    else:
        image_key, label_key = "images", "labels"
    trainset = datasets.MNIST(root='/home/chenhao/caps3d/data/MNIST_data', train=True, download=True)
    mnist_train = {}
    mnist_train["images"] = trainset.train_data.numpy()
    mnist_train["labels"] = trainset.train_labels.numpy()
        # im = dataset['train']['images'][index]
        # mlab.mesh(x, y, z, scalars=im, colormap="coolwarm")
        # print(f"This image is {label.item()}")

        #for n in range(10):
    n = 22
    print(n)
    im = mnist_train[image_key][n]
    label = mnist_train[label_key][n]
    im = np.array(im, dtype=np.dtype("uint8"))
    print(label)
    filename="mnist_train_"+"n_"+str(n)+"_lable_"+str(label)+".png"
    Image.fromarray(im).save(filename)
    plt.imshow(im,cmap ='gray')
    plt.show()



if __name__ == "__main__":
    #plot_smnist()
    #plot_smnist(True)
    plot_mnist(True)
    # paths = glob.glob("data/modelnet40/modelnet40_train/*.off")
    # for path in paths:
    #     print(path)
    #     scene = with_preprocess(path, b=8, k=1)
    #     scene.show()

    # path="data/modelnet40/modelnet40_train/guitar_0089.off"
    # scene = with_preprocess(path, b=8, k=1)
    # scene.show()
