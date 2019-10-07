import numpy as np
import constants

import numpy as np
import random
import math

import matplotlib

import numpy.random as nprand
import time
import scipy
import scipy.interpolate

from contours.core import shapely_formatter as shapely_fmt
from contours.quad import QuadContourGenerator

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors


BIG_NUMBER = 1e8


def interiorPoint(dist, cutoff):
    ret = np.zeros(shape=dist.shape, dtype=bool)
    ret[dist <= 0] = True
    return ret


def envelopePoint(dist, cutoff):
    ret = np.zeros(shape=dist.shape, dtype=bool)
    ret[np.abs(dist) <= cutoff / 2] = True
    return ret


def computeInsideOutsideCells(dist, func, cutoff):
    d_inside = func(dist, cutoff)
    d_outside = np.logical_not(d_inside)

    BL_inside = d_inside[:-1, : -1]
    BR_inside = d_inside[1:, : -1]
    TL_inside = d_inside[:-1, 1:]
    TR_inside = d_inside[1:, 1:]

    cell_inside = BL_inside & BR_inside & TL_inside & TR_inside

    BL_outside = d_outside[:-1, : -1]
    BR_outside = d_outside[1:, : -1]
    TL_outside = d_outside[:-1, 1:]
    TR_outside = d_outside[1:, 1:]

    cell_outside = BL_outside & BR_outside & TL_outside & TR_outside
    return cell_inside, cell_outside


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x


def padArray(arr, newshape, xoff, yoff):
    result = np.zeros(newshape)
    result[xoff:arr.shape[0] + xoff, yoff:arr.shape[1] + yoff] = arr
    return result


eps = 1e-4


class LevelSet:
    def __init__(self, Nx, Ny, Lx, Ly, init_conds):

        self.Nx = Nx
        self.Ny = Ny

        self.Lx = Lx
        self.Ly = Ly

        self.delta_x = self.Lx / self.Nx
        self.delta_y = self.Ly / self.Ny

        self.xpts = np.linspace(-self.Lx / 2., self.Lx / 2., self.Nx + 1)
        self.ypts = np.linspace(-self.Ly / 2., self.Ly / 2., self.Ny + 1)

        X, Y = np.meshgrid(self.xpts, self.ypts)

        self.X = X
        self.Y = Y
        self.SDF = init_conds.computeMinDistance(X, Y)
        # outside is
        # self.SDF[np.sqrt(self.X**2 + self.Y**2) > constants.R_outer] = BIG_NUMBER

        self.env_width = 0.0002

        self.early_detect_cells = True

        # make sure envelope can't be contained entirely in a cell
        self.cutoff_env_width = 1.5 * max(self.delta_x, self.delta_y)

        self.num_vox_mc_pts = 100

        self.setStale()

    def computeUpwindGrad(self):
        # compute element differences
        diffX = np.diff(self.SDF, axis=0)
        diffY = np.diff(self.SDF, axis=1)

        # compute forward and backward differences
        diffX_p = padArray(diffX, self.SDF.shape, 0, 0)
        diffX_m = padArray(diffX, self.SDF.shape, 1, 0)

        diffY_p = padArray(diffY, self.SDF.shape, 0, 0)
        diffY_m = padArray(diffY, self.SDF.shape, 0, 1)

        under_rad = (np.maximum(diffX_m, 0)**2 + np.minimum(diffX_p, 0)**2) / (self.delta_x ** 2) + \
            (np.maximum(diffY_m, 0)**2 + np.minimum(diffY_p, 0)**2) / (self.delta_y ** 2)

        return np.sqrt(under_rad)

    def advance(self, vel):
        cfl_fac = np.abs(vel * constants.delta_t / self.delta_x)

        assert(cfl_fac <= 1.0)

        # compute gradient
        gradNorm = self.computeUpwindGrad()

        # enforce homogeneous neumann BC
        gradNorm[np.sqrt(self.X**2 + self.Y**2) >= constants.R_outer] = 0
        delta = -1.0 * gradNorm * vel * constants.delta_t

        # apply update
        self.SDF = self.SDF + delta

        self.setStale()

    def getSDF(self):
        return self.SDF

    def getGrid(self):
        return self.X, self.Y

    def setStale(self):
        # print("Setting stale")
        self.cell_inside = None
        self.cell_outside = None
        self.zerocontour = None

    def MCIntegrate(self, func, refine_inside=False):
        if self.cell_inside is None:
            self.cell_inside, self.cell_outside = computeInsideOutsideCells(
                self.SDF, func, self.cutoff_env_width)

        num_inside = 0

        dxt, dyt = nprand.random((2, self.num_vox_mc_pts))

        for idy in range(self.Ny):
            for idx in range(self.Nx):
                if self.early_detect_cells:
                    # reject if all corners outside
                    if self.cell_outside[idx, idy]:
                        continue
                    if self.cell_inside[idx, idy]:
                        if not refine_inside:
                            num_inside += self.num_vox_mc_pts
                            continue

                dxb = 1 - dxt
                dyb = 1 - dyt

                pts = (self.SDF[idx, idy],
                       self.SDF[idx, idy + 1],
                       self.SDF[idx + 1, idy],
                       self.SDF[idx + 1, idy + 1])

                weights = np.array((dxb * dyb,
                                    dxb * dyt,
                                    dxt * dyb,
                                    dxt * dyt))

                dists = weights.T.dot(pts)
                num_inside += np.count_nonzero(func(dists, self.env_width))

        return num_inside

    def getZeroContour(self):
        SDF = np.copy(self.SDF)
        # compute contour filled from -inf to zero (in shapely)
        if self.zerocontour is None:
            c = QuadContourGenerator.from_rectilinear(self.xpts, self.ypts, SDF, shapely_fmt)
            # print(c.filled_contour(max=0.0), type(c.filled_contour(max=0.0)), len(c.filled_contour(max=0.0)))
            poly = c.filled_contour(max=0.0)[0]

            nm = np.max(SDF[self.X**2 + self.Y**2 <= constants.R_outer**2])

            min_dist = min(self.delta_x, self.delta_y)

            if nm <= min_dist * 2:
                # smooth polygon if close to end
                poly = poly.buffer(1, resolution = 16, join_style=1).buffer(-1, resolution = 16, join_style=1)

            self.zerocontour = poly

        return self.zerocontour

    def computeArea(self):
        return self.getZeroContour().area

    def computePerim(self):
        return self.getZeroContour().length

    def getDomainArea(self):
        return self.Lx * self.Ly

    def make_plot(self, title_str=""):
        SDF = np.copy(self.SDF)
        fig, ax = plt.subplots()

        inside_vals = np.where(SDF <= 0)
        BC_vals = np.where(self.X**2 + self.Y**2 > constants.R_outer ** 2)

        SDF[BC_vals] = BIG_NUMBER
        SDF[inside_vals] = np.nan
        SDF_mid = midpoints(SDF)

        sdfplot = ax.imshow(SDF_mid,
                            cmap='coolwarm_r', vmax=constants.R_outer, extent=[-self.Lx / 2., self.Lx / 2., -self.Ly / 2., self.Ly / 2.])

        SDF_mid[:] = 0

        cMap = colors.ListedColormap(['slategray'])

        ax.set_title("Level set SRM burn {}".format(title_str))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        poly = self.getZeroContour()

        if poly.type == 'Polygon':
            ax.plot(*poly.exterior.xy, "k")
        elif poly.type == 'MultiPolygon':
            for p in poly:
                ax.plot(*p.exterior.xy, "k")

        cb = fig.colorbar(sdfplot, ax=ax)
        cb.set_label("Distance from burn front")
        return fig
