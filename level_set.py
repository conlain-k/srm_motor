import numpy as np
import constants

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

import numba

from enum import Enum


class GradientMode(Enum):
    UPWIND_FIRST = 1
    UPWIND_SECOND = 2
    # this is a terrible method
    CENTERED_DIFF = 69


BIG_NUMBER = 1e8


def _WENO_comp(d1, d2, d3, d4, d5):
    phi1 = (d1 / 3.) - (7. * d2 / 6.) + (11. * d3 / 6.)
    phi2 = (-d2 / 6.) + (5. * d3 / 6) + (d4 / 3.)
    phi3 = (d3 / 3.) + (5. * d4 / 6.) - (d5 / 6.)

    S1 = (13. / 12.) * (d1 - 2 * d2 + d3) ** 2 + (1. / 4.) * (d1 - 4 * d2 + 3 * d3)**2
    S2 = (13. / 12.) * (d2 - 2 * d3 + d4) ** 2 + (1. / 4.) * (d2 - d4)**2
    S3 = (13. / 12.) * (d3 - 2 * d4 + d5) ** 2 + (1. / 4.) * (3 * d3 - 4 * d4 + d5)**2

    # make epsilon represent largest difference
    eps = 1e-6 * max([d1**2, d2**2, d3**2, d4**2, d5**2]) + 1e-99

    alpha1 = 1 / (S1 + eps)**2
    alpha2 = 6 / (S2 + eps)**2
    alpha3 = 3 / (S3 + eps)**2

    alpha_sum = alpha1 + alpha2 + alpha3
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum
    w3 = alpha3 / alpha_sum

    return w1 * phi1 + w2 * phi2 + w3 * phi3


@numba.stencil
def _WENO_stencil(SDF, dx):
    # backward stencil
    d1_m = (SDF[0, i - 2] - SDF[0, i - 3]) / dx
    d2_m = (SDF[0, i - 1] - SDF[0, i - 2]) / dx
    d3_m = (SDF[0, i + 0] - SDF[0, i - 1]) / dx
    d4_m = (SDF[0, i + 1] - SDF[0, i + 0]) / dx
    d5_m = (SDF[0, i + 2] - SDF[0, i + 1]) / dx

    Dm = _WENO_comp(d1_m, d2_m, d3_m, d4_m, d5_m)

    d1_p = (SDF[0, i + 3] - SDF[0, i + 2]) / dx
    d2_p = (SDF[0, i + 2] - SDF[0, i + 1]) / dx
    d3_p = (SDF[0, i + 1] - SDF[0, i + 0]) / dx
    d4_p = (SDF[0, i + 0] - SDF[0, i - 1]) / dx
    d5_p = (SDF[0, i - 1] - SDF[0, i - 2]) / dx

    Dp = _WENO_comp(d1_p, d2_p, d3_p, d4_p, d5_p)


@numba.jit
def _switch(x, y):
    # local extrema => zero slope
    if x * y <= 0:
        return 0

    # return smaller (slope limiter)
    if abs(x) <= abs(y):
        return x

    return y


@numba.stencil
def _upwindSecond(SDF, dX):
    diffX_mm = (2. * SDF[0, 0] - 3. * SDF[0, -1] + SDF[0, -2]) / dX
    diffX_pm = (SDF[0, 1] - 2. * SDF[0, 0] + SDF[0, -1]) / dX
    diffX_pp = (2. * SDF[0, 2] - 3. * SDF[0, 1] + SDF[0, 0]) / dX

    diffX_b = 0.5 * _switch(diffX_mm, diffX_pm) + (SDF[0, 0] - SDF[0, -1]) / dX
    diffX_f = 0.5 * _switch(diffX_pp, diffX_pm) + (SDF[0, 1] - SDF[0, 0]) / dX

    return np.minimum(diffX_f, 0)**2 + np.maximum(diffX_b, 0)**2


@numba.jit
def upwindGradSecond(SDF, dx, dy):
    Dx2 = _upwindSecond(SDF, dx)
    Dy2 = _upwindSecond(SDF.T, dy)
    return np.sqrt(Dx2 + Dy2)


@numba.stencil
def _upwindFirst(SDF, dX):
    diffX_f = (SDF[0, 1] - SDF[0, 0]) / dX
    diffX_b = (SDF[0, 0] - SDF[0, -1]) / dX
    diff_X2 = np.minimum(diffX_f, 0)**2 + np.maximum(diffX_b, 0)**2
    return diff_X2


@numba.jit
def upwindGradFirst(SDF_, dx, dy):
    SDF = np.copy(SDF_)
    pd = 1

    Dx2 = _upwindFirst(SDF_, dx)
    Dy2 = _upwindFirst(SDF_.T, dy)
    reg_grad = np.sqrt(Dx2 + Dy2)

    return reg_grad


def centerGrad(SDF_, dx, dy):
    gx, gy = np.gradient(SDF_,  [dx, dy])
    return np.sqrt(gx * gx + gy * gy)


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x


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
        # self.SDF[self.getOutsidePoints()] = 1e4

        self.setStale()
        self.setGradientMode(GradientMode.UPWIND_FIRST)

    def reinitialize(self):
        old_sign = np.sign(np.copy(self.SDF))
        envelope_points = np.abs(self.SDF) <= constants.R_outer / 10

        def iterate():
            print("iter")
            gradNorm = self.computeGrad()
            delta = -1. * old_sign * (gradNorm - 1.)
            self.SDF[envelope_points] += delta[envelope_points] * constants.delta_t
            print(delta[envelope_points], np.max(delta[envelope_points]))
            return np.any(np.abs(delta[envelope_points]) >= 0.25)

        self.make_plot(thing_to_plot=self.computeGrad())
        self.make_plot(thing_to_plot=old_sign)

        while iterate():
            pass
        # [iterate() for i     in range(20)]

        self.make_plot(thing_to_plot=self.computeGrad())

    def getOutsidePoints(self):
        return np.where(np.sqrt(self.X**2 + self.Y**2) >= constants.R_outer)

    def getInsidePoints(self):
        return np.where(np.sqrt(self.X**2 + self.Y**2) < constants.R_outer)

    def setGradientMode(self, mode):
        assert(mode is not None)
        self.gradient_mode = mode

    def computeUpwindGrad(self):
        # compute element differences
        return upwindGradSecond(self.SDF, self.delta_x, self.delta_y)

    def computeGrad(self, mode=None):
        if mode is None:
            mode = self.gradient_mode
        grad = None
        if mode == GradientMode.UPWIND_FIRST:
            grad = upwindGradFirst(self.SDF, self.delta_x, self.delta_y)
        elif mode == GradientMode.UPWIND_SECOND:
            grad = upwindGradSecond(self.SDF, self.delta_x, self.delta_y)

        assert(grad is not None)

        return grad

    def advance(self, vel):
        cfl_fac = np.abs(vel * constants.delta_t / self.delta_x)

        assert(cfl_fac <= 1.0)

        # compute gradient
        gradNorm = self.computeGrad()

        # enforce homogeneous neumann BC
        gradNorm[self.getOutsidePoints()] = 0

        # change to level set update
        delta = -1.0 * gradNorm * vel * constants.delta_t

        # apply update
        self.SDF += delta

        self.setStale()

    def getSDF(self):
        return self.SDF

    def getGrid(self):
        return self.X, self.Y

    def setStale(self):
        self.zerocontour = None

    def getZeroContour(self):
        SDF = np.copy(self.SDF)
        # compute contour filled from -inf to zero (in shapely)
        if self.zerocontour is None:
            c = QuadContourGenerator.from_rectilinear(self.xpts, self.ypts, SDF, shapely_fmt)
            # print(c.filled_contour(max=0.0), type(c.filled_contour(max=0.0)), len(c.filled_contour(max=0.0)))
            cutoff_len = 0
            poly = c.filled_contour(max=cutoff_len)[0]

            nm = np.max(SDF[self.getInsidePoints()])

            min_dist = min(self.delta_x, self.delta_y)

            smooth_at_end = True
            if smooth_at_end and nm <= min_dist * 2:
                # smooth polygon if close to end
                poly = poly.buffer(1, resolution=16, join_style=1).buffer(-1,
                                                                          resolution=16, join_style=1)

            self.zerocontour = poly

        return self.zerocontour

    def computeArea(self):
        return self.getZeroContour().area

    def computePerim(self):
        return self.getZeroContour().length

    def getDomainArea(self):
        return self.Lx * self.Ly

    def make_plot(self, title_str="", thing_to_plot=None, show_zero_set = True):
        if thing_to_plot is None:
            SDF = np.copy(self.SDF)
        else:
            SDF = np.copy(thing_to_plot)

        fig, ax = plt.subplots()

        burned_val = np.where(SDF <= 0)
        exterior_vals = np.where(self.X**2 + self.Y**2 >= constants.R_outer ** 2)

        SDF[exterior_vals] = BIG_NUMBER
        SDF[burned_val] = np.nan
        SDF_mid = SDF

        sdfplot = ax.imshow(SDF_mid,
                            cmap='coolwarm_r', vmax=constants.R_outer, vmin=0,
                            extent=[-self.Lx / 2., self.Lx / 2., -self.Ly / 2., self.Ly / 2.], origin='lower')
        cb = fig.colorbar(sdfplot, ax=ax)
        cb.set_label("Distance from burn front (in)")

        ax.set_title("Level set SRM burn {}".format(title_str))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if show_zero_set:
            poly = self.getZeroContour()
            
            if poly.type == 'Polygon':
                ax.plot(*poly.exterior.xy, "k")
            elif poly.type == 'MultiPolygon':
                for p in poly:
                    ax.plot(*p.exterior.xy, "k")

        return fig
