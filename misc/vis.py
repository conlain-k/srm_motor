"""
============
3D animation
============

A simple example of an animated plot... In 3D!
"""
from matplotlib.colors import to_rgba
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
# matplotlib.use('Qt4Agg')


def Gen_RandLine(length, dims=2):
    """
    Create a line using a random walk algorithm

    length is the number of points for the line.
    dims is the number of dimensions the line has.
    """
    lineData = np.empty((dims, length))
    lineData[:, 0] = np.random.rand(dims)
    for index in range(1, length):
        # scaling the random numbers by 0.1 so
        # movement is small compared to position.
        # subtraction by 0.5 is to change the range to [-0.5, 0.5]
        # to allow a line to move backwards.
        step = ((np.random.rand(dims) - 0.5) * 0.1)
        lineData[:, index] = lineData[:, index - 1] + step

    return lineData


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines


def genPts(rad_outer, half_length):

    Lx, Ly, Lz = rad_outer + 1, half_length + 1, rad_outer + 1
    Nx, Ny, Nz = 20, 20, 20

    X, Y, Z = np.meshgrid(np.linspace(-Lx, Lx, Nx),
                          np.linspace(-Ly, Ly, Ny), np.linspace(-Lz, Lz, Nz))

    return X, Y, Z


def setCylVals(X, Y, Z, rad_inner, rad_outer, half_length):
    R = np.sqrt(X**2 + Z**2)
    inside = (R <= rad_outer) & (R >= rad_inner) & (Y <= half_length)

    return inside


def getColor(X, Y, Z, dist):

    maxdist = np.max(dist.ravel())

    inside_dist = dist[dist >= 0]

    X_inside = X[dist >= 0]
    Y_inside = Y[dist >= 0]
    Z_inside = Z[dist >= 0]

    dist_scaled = inside_dist / maxdist

    colors = np.zeros(inside_dist.shape + (3,))

    print(inside_dist.shape, colors.shape, colors[0].shape)

    colors[:, 0] = np.abs(X_inside / np.max(X_inside.ravel()))
    colors[:, 1] = np.abs(Y_inside / np.max(Y_inside.ravel()))
    colors[:, 2] = np.abs(Z_inside / np.max(Z_inside.ravel()))
    # colors[inside_domain, 3] = 0.5

    return colors


def setCylSDF(X, Y, Z, rad_inner, rad_outer, half_length):
    R = np.sqrt(X**2 + Z**2)
    inside = (R <= rad_outer) & (R >= rad_inner) & (Y <= half_length)

    dist_cap = Y - half_length
    dist_cap[Y < 0] = np.abs(Y[Y < 0]) - half_length
    # flip sign to have inwards normal off cap
    dist_cap = -1. * dist_cap

    # distance from inner cyl, positive if outside inner cyl (inside domain)
    dist_inside = R - rad_inner

    dist_total = np.minimum(dist_inside, dist_cap)

    mask = np.ones_like(dist_total, dtype=bool)
    mask[inside] = False

    # set outside values to None
    dist_total[mask] = None

    return dist_total


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x


def create3DFigure():
        # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('3D Test')
    return fig, ax


def computeDists(X, Y, Z, fig, ax, rad_inner, rad_outer, length):
    midX = midpoints(X)
    midY = midpoints(Y)
    midZ = midpoints(Z)

    cyl_pts = setCylVals(midX, midY, midZ, rad_inner, rad_outer, length / 2.)
    cyl_dist = setCylSDF(midX, midY, midZ, rad_inner, rad_outer, length / 2.)
    return cyl_dist


def plotDists(X, Y, Z, ax, cyl_dist):
    cmap = plt.get_cmap('seismic_r')

    mapped_colors = cmap(cyl_dist)

    mapped_colors[..., 3] = 0.8
    ax.voxels(X, Y, Z, filled=(cyl_dist > 0), facecolors=mapped_colors, edgecolors=[1., 1., 1., .2])


def plotCylBurn(rad_outer, init_length, r_vals, L_vals):
    fig, ax = create3DFigure()

    X, Y, Z = genPts(rad_outer, init_length / 2.)

    cyl_dist = computeDists(X, Y, Z, fig, ax, r_vals[0], rad_outer, L_vals[0])

    def animate(i):
        ind = i * 10
        print("Showing timestep {}".format(ind))
        cyl_dist = computeDists(X, Y, Z, fig, ax, r_vals[ind], rad_outer, L_vals[ind])
        plotDists(X, Y, Z, ax, cyl_dist)

    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=r_vals.size // 10)
    plt.show()


if __name__ == "__main__":
    rad_inner = 1.
    rad_outer = 2.
    length = 4.

    fig, ax = create3DFigure()
    X, Y, Z = genPts(rad_outer, length / 2.)

    dists = computeDists(X, Y, Z, fig, ax, rad_inner, rad_outer, length)
    plotDists(X, Y, Z, ax, dists)

    plt.show()
