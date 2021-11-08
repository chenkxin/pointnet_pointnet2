import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab


def plot_sgrid(s):
    # learn from https://matplotlib.org/gallery/mplot3d/scatter3d.html#sphx-glr-gallery-mplot3d-scatter3d-py
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    step = 2
    ax.scatter(s[:, 0], s[:, 1], s[:, 2])
    plt.show()


def test_plot_sgrid():
    # create 64*64 points on a sphere
    from lib.dataset.transforms import make_sgrid

    s = make_sgrid(32, 0, 0, 0)
    plot_sgrid(s)


def plot_grid():
    # Make sphere, choose colors
    # b=8j
    # phi, theta = np.mgrid[0:np.pi:b, 0:2 * np.pi:b]
    # x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)

    b = 4

    # shperical signal
    from lib.dataset.transforms import make_sgrid

    s = make_sgrid(b, 0, 0, 0)
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    mlab.points3d(x, y, z, scale_factor=0.1)

    # ring
    from s2cnn import s2_equatorial_grid

    grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * b, n_beta=1)
    grid = np.array(grid)

    phi, theta = grid[:, 0], grid[:, 1]
    x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
    mlab.points3d(x, y, z, color=(1, 0, 0), scale_factor=0.15)

    # rotate the ring by some angle
    from lib.dataset.transforms import rotmat

    R = rotmat(0, np.pi / 4, 0, hom_coord=False)
    grid = np.array([x, y, z]).reshape((3, 8))
    grid = np.dot(R, grid)
    # grid = np.concatenate((grid, np.ones(8).reshape(1, 8)), axis=0)
    # grid = np.dot(R, grid)
    # grid = grid.reshape(-1,4)[:,:3]
    # grid = np.einsum('ij,nj->ni', R, grid)
    x, y, z = grid[0, :], grid[1, :], grid[2, :]
    mlab.points3d(x, y, z, color=(0, 1, 0), scale_factor=0.2)

    mlab.view()
    mlab.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x,y,z)
    # plt.show()


def spherical_example():
    # https://stackoverflow.com/questions/23517416/colors-on-a-sphere-to-depict-values

    # Make sphere, choose colors
    phi, theta = np.mgrid[0 : np.pi : 64j, 0 : 2 * np.pi : 64j]

    x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
    s = x * y  # <-- colors

    # Display
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 500))
    mlab.mesh(x, y, z, scalars=s, colormap="Spectral")
    mlab.view()
    mlab.show()


def spherical_example1():
    # Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
    # Copyright (c) 2008, Enthought, Inc.
    # License: BSD Style.

    import numpy as np
    from mayavi import mlab
    from scipy.special import sph_harm

    # Create a sphere
    r = 0.3
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:101j, 0 : 2 * pi : 101j]

    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
    mlab.clf()
    # Represent spherical harmonics on the surface of the sphere
    for n in range(1, 6):
        for m in range(n):
            s = sph_harm(m, n, theta, phi).real

            mlab.mesh(x - m, y - n, z, scalars=s, colormap="jet")

            s[s < 0] *= 0.97

            s /= s.max()
            mlab.mesh(s * x - m, s * y - n, s * z + 1.3, scalars=s, colormap="Spectral")

    mlab.view(90, 70, 6.2, (-1.3, -2.9, 0.25))
    mlab.show()


def plot_3d_spherical():
    # plot 10 3d spherical example
    import numpy as np
    from lib.dataset import makeDataset
    from lib.dataset.transforms import make_sgrid

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1000, 800))

    # Make sphere, choose colors
    phi, theta = np.mgrid[0 : np.pi : 64j, 0 : 2 * np.pi : 64j]
    x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
    dataset = makeDataset(
        train=True,
        b=32,
        dataset_name="modelnet40",
        root="~/caps3d/data",
        normalize=False,
    )
    for n in range(10):
        f = dataset[n * 100][0]
        f = f[0, :, :]
        mlab.mesh(x, y + n * 2 + 0.2, z, scalars=f, colormap="coolwarm")

    mlab.view()
    mlab.show()


def plot_3d_spherical1():
    import glob

    from lib.dataset.data_viz import with_preprocess

    paths = glob.glob("/home/qiangzibro/caps3d/data/modelnet40/modelnet40_train/*.off")
    for path in paths:
        mesh = with_preprocess(path)


def plot_kernel():
    # https://stackoverflow.com/questions/23517416/colors-on-a-sphere-to-depict-values
    import numpy as np
    from lib.dataset.transforms import make_sgrid
    from mayavi import mlab
    from s2cnn.s2_grid import s2_near_identity_grid

    s = np.array(s2_near_identity_grid())
    # Make sphere, choose colors
    phi, theta = s[:, 0], s[:, 1]

    x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, cmap=plt.get_cmap("CMRmap_r"), s=50)

    s = make_sgrid(32, 0, 0, 0)
    # ax.scatter(s[:, 0], s[:, 1], s[:, 2],cmap=plt.get_cmap("Accent"))

    plt.show()


if __name__ == "__main__":
    plot_3d_spherical()
