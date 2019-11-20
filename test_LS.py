import level_set
import srm_system
import shapes
import constants

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors

import numpy as np
import numba


def testArea():
    rad = constants.R_inner
    L = 4 * rad + 2

    def computeCircVals(num, show_plot=False):
        # make circle
        shapecoll = shapes.ShapeCollection()
        circ = shapes.Circle((1, 1), rad)
        shapecoll.addShape(circ)
        circ_ls = level_set.LevelSet(num, num, L, L, shapecoll)
        if show_plot:
            circ_ls.make_plot()

        circ_area = circ_ls.computeArea()
        circ_perim = circ_ls.computePerim()

        gradNorm = circ_ls.computeUpwindGrad()

        return circ_area, circ_perim

    N = np.arange(4, 51, 2)
    A = np.zeros_like(N, dtype=np.float64)
    P = np.zeros_like(N, dtype=np.float64)
    for ind, num in enumerate(N):
        A[ind], P[ind] = computeCircVals(num)

    actual_area = np.pi * rad * rad
    actual_perim = 2 * np.pi * rad

    A_err = (actual_area - A) / actual_area
    P_err = (actual_perim - P) / actual_perim

    print("acc at end is {}".format(1 - A_err[-1]))

    plt.figure()
    plt.plot(N, 1 - A_err, label='area')
    plt.plot(N, 1 - P_err, label='perim')
    plt.legend()
    plt.grid()
    plt.xlabel("Nx = Ny")
    plt.xticks(np.arange(0, 51, 10))
    plt.ylabel("Accuracy (1 - % err)")

    ymin, ymax = plt.ylim()
    ymax = max(1, ymax)
    # ymin = min(0, ymin)
    plt.title("Area of circle reconstruction from grid points")
    plt.ylim(ymin, ymax)
    plt.show()

def testGrad():
    N = 50
    rad = constants.R_inner
    L = 4 * rad + 2

    shapecoll = shapes.ShapeCollection()
    circ1 = shapes.Circle((0.5, 0), rad)
    circ2 = shapes.Circle((-0.5, 0), rad)
    shapecoll.addShape(circ1)
    shapecoll.addShape(circ2)
    circ_ls = level_set.LevelSet(N, N, L, L, shapecoll)

    SDF = circ_ls.getSDF()

    circ_ls.reinitialize()

    grad1 = circ_ls.computeGrad(level_set.GradientMode.UPWIND_FIRST)
    # grad2 = circ_ls.computeGrad(level_set.GradientMode.UPWIND_SECOND)

    for i in range(100):
        circ_ls.advance(1)
        grad = circ_ls.computeGrad()
        print(np.max(grad[circ_ls.getInsidePoints()]))
        # circ_ls.make_plot(thing_to_plot = grad)

    grad_final = circ_ls.computeGrad()

    circ_ls.make_plot(thing_to_plot = grad_final)
    circ_ls.make_plot()

    plt.show()
    # 
    # axes[0].set(aspect='equal')
    # axes[1].set(aspect='equal')
    # axes[2].set(aspect='equal')

    plt.show()


def main():
    testGrad()
    testArea()


if __name__ == "__main__":
    main()
