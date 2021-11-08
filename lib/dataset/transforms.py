"""
3D model transforms
*.off 3d file --> ToMesh --> ProjectOnSphere --> CacheNPY

also some util functions provided
"""
import logging
import os

import numpy as np
import torchvision
import trimesh

from utils import load_mean_std

np.seterr(divide="ignore", invalid="ignore")
logging.getLogger("pyembree").disabled = True


def get_transform(
    dataset_name,
    train,
    bandwidth,
    normalize,
    repeat=1,
    type="original",
    rot=False,
    **kwargs
):
    transforms = [
        ToMesh(random_rotations=rot, random_translation=0),
        ProjectOnSphere(
            bandwidth=bandwidth,
            dataset=dataset_name,
            train=train,
            normalize=normalize,
        ),
    ]
    return CacheNPY(
        prefix="b{}_".format(bandwidth),
        repeat=repeat,
        transform=torchvision.transforms.Compose(transforms=transforms),
        pick_randomly=False,
        type=type,
        **kwargs
    )


class ToMesh:
    def __init__(self, random_rotations=False, random_translation=0):
        self.rot = random_rotations
        self.tr = random_translation

    def __call__(self, path):
        mesh = trimesh.load_mesh(path)
        mesh.remove_degenerate_faces()
        mesh.fix_normals()
        mesh.fill_holes()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()

        mesh.apply_translation(-mesh.centroid)

        r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
        mesh.apply_scale(1 / r)

        if self.tr > 0:
            tr = np.random.rand() * self.tr
            rot = rnd_rot()
            mesh.apply_transform(rot)
            mesh.apply_translation([tr, 0, 0])

            if not self.rot:
                mesh.apply_transform(rot.T)

        if self.rot:
            mesh.apply_transform(rnd_rot())

        r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
        mesh.apply_scale(0.99 / r)

        return mesh

    def __repr__(self):
        return self.__class__.__name__ + "(rotation={0}, translation={1})".format(
            self.rot, self.tr
        )


class ProjectOnSphere:
    def __init__(self, bandwidth, dataset, train=True, normalize=True):
        self.train = train
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.dataset = dataset
        self.sgrid = make_sgrid(bandwidth, alpha=0, beta=0, gamma=0)  # grid on sphere

    def __call__(self, mesh):
        im = render_model(mesh, self.sgrid)  # shape 3_channels x #v
        im = im.reshape(3, 2 * self.bandwidth, 2 * self.bandwidth)
        from scipy.spatial.qhull import QhullError  # pylint: disable=E0611

        try:
            convex_hull = mesh.convex_hull
        except QhullError:
            convex_hull = mesh

        hull_im = render_model(convex_hull, self.sgrid)
        hull_im = hull_im.reshape(3, 2 * self.bandwidth, 2 * self.bandwidth)
        im = np.concatenate([im, hull_im], axis=0)
        assert len(im) == 6

        # take absolute value of normal
        im[1] = np.absolute(im[1])
        im[4] = np.absolute(im[4])

        # TODO

        if self.normalize:
            mean, std = load_mean_std(dataset_name=self.dataset, train=self.train)
            im = im - mean
            im = im / std
        im = im.astype(np.float32)  # pylint: disable=E1101

        return im

    def __repr__(self):
        return self.__class__.__name__ + "(bandwidth={0})".format(self.bandwidth)


class CacheNPY(object):
    def __init__(
        self, prefix, repeat, transform, pick_randomly=True, type="original", **kwargs
    ):
        self.transform = transform
        self.prefix = prefix
        self.repeat = repeat
        self.pick_randomly = pick_randomly
        self.type = type

    def check_trans(self, file_path):
        #  print("transform {}...".format(file_path))
        try:
            return self.transform(file_path)
        except Exception as e:
            print("Exception during transform of {}".format(file_path))
            raise e

    def _head(self, dir):
        path = os.path.join(dir, self.type)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def __call__(self, file_path):
        head, tail = os.path.split(file_path)
        head = self._head(head)
        root, _ = os.path.splitext(tail)
        npy_path = os.path.join(head, self.prefix + "r{}_" + root + ".npy")

        exists = [os.path.exists(npy_path.format(i)) for i in range(self.repeat)]

        if self.pick_randomly and all(exists):
            i = np.random.randint(self.repeat)
            try:
                return np.load(npy_path.format(i))
            except OSError:
                exists[i] = False

        if self.pick_randomly:
            img = self.check_trans(file_path)
            np.save(npy_path.format(exists.index(False)), img)

            return img

        output = []
        for i in range(self.repeat):
            try:
                img = np.load(npy_path.format(i))
            except (OSError, FileNotFoundError):
                img = self.check_trans(file_path)
                np.save(npy_path.format(i), img)
            output.append(img)

        return output

    def __repr__(self):
        return self.__class__.__name__ + "(prefix={0}, transform={1})".format(
            self.prefix, self.transform
        )


def readlines(path):
    fo = open(path)
    next(fo)
    filelist = fo.readlines()
    numberoflines = len(filelist)
    returnMat = np.zeros((numberoflines, 6))
    index = 0
    for line in filelist:
        line = line.strip()
        listline = line.split(" ")
        returnMat[index, :] = listline[0:6]
        index += 1
    fo.close()
    return returnMat


def rotmat(
    a, b, c, hom_coord=False
):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
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


def make_sgrid(b, alpha, beta, gamma):
    from lie_learn.spaces import S2

    theta, phi = S2.meshgrid(b=b, grid_type="SOFT")
    sgrid = S2.change_coordinates(
        np.c_[theta[..., None], phi[..., None]], p_from="S", p_to="C"
    )
    sgrid = sgrid.reshape((-1, 3))

    R = rotmat(alpha, beta, gamma, hom_coord=False)
    sgrid = np.einsum("ij,nj->ni", R, sgrid)  # sgrid.dot(R')
    return sgrid


def render_model(mesh, sgrid):
    # Cast rays
    # triangle_indices = mesh.ray.intersects_first(ray_origins=sgrid, ray_directions=-sgrid)
    index_tri, index_ray, loc = mesh.ray.intersects_id(
        ray_origins=sgrid,
        ray_directions=-sgrid,
        multiple_hits=False,
        return_locations=True,
    )
    loc = loc.reshape((-1, 3))  # fix bug if loc is empty
    # assert loc.shape[0]==4096
    # Each ray is in 1-to-1 correspondence with a grid point. Find the position of these points
    grid_hits = sgrid[index_ray]
    grid_hits_normalized = grid_hits / np.linalg.norm(grid_hits, axis=1, keepdims=True)

    # Compute the distance from the grid points to the intersection pionts
    dist = np.linalg.norm(grid_hits - loc, axis=-1)

    # For each intersection, look up the normal of the triangle that was hit
    normals = mesh.face_normals[index_tri]
    normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Construct spherical images
    dist_im = np.ones(sgrid.shape[0])
    dist_im[index_ray] = dist
    # dist_im = dist_im.reshape(theta.shape)

    # shaded_im = np.zeros(sgrid.shape[0])
    # shaded_im[index_ray] = normals.dot(light_dir)
    # shaded_im = shaded_im.reshape(theta.shape) + 0.4

    n_dot_ray_im = np.zeros(sgrid.shape[0])
    # n_dot_ray_im[index_ray] = np.abs(np.einsum("ij,ij->i", normals, grid_hits_normalized))
    n_dot_ray_im[index_ray] = np.einsum(
        "ij,ij->i", normalized_normals, grid_hits_normalized
    )

    nx, ny, nz = (
        normalized_normals[:, 0],
        normalized_normals[:, 1],
        normalized_normals[:, 2],
    )
    gx, gy, gz = (
        grid_hits_normalized[:, 0],
        grid_hits_normalized[:, 1],
        grid_hits_normalized[:, 2],
    )
    wedge_norm = np.sqrt(
        (nx * gy - ny * gx) ** 2 + (nx * gz - nz * gx) ** 2 + (ny * gz - nz * gy) ** 2
    )
    n_wedge_ray_im = np.zeros(sgrid.shape[0])
    n_wedge_ray_im[index_ray] = wedge_norm

    # Combine channels to construct final image
    # im = dist_im.reshape((1,) + dist_im.shape)
    im = np.stack((dist_im, n_dot_ray_im, n_wedge_ray_im), axis=0)
    # print(im.shape)
    return im


def rnd_rot():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, np.arccos(z), c, True)
    return rot
