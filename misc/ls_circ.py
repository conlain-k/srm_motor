import numpy as np
import numpy.random as nprand
import time
import scipy
import scipy.interpolate
from cProfile import Profile
from pstats import Stats

from matplotlib import pyplot as plt

from ../srm_system import *
import ../shapes
from ../constants import *
prof = Profile()

prof.disable()  # don't time setup


shapecoll = shapes.ShapeCollection()
circ = shapes.Circle((0, 0), R_inner)
shapecoll.addShape(circ)

SRM = LevelSetSRM(shapecoll)

LS = SRM.level_set


def computeArea():

    return LS.computeArea()


def computePerim():

    return LS.computePerim()


def main():
    actual_area = R_inner * R_inner * np.pi
    actual_perim = 2. * np.pi * R_inner
    prof.enable()
    times = []
    areas = []
    perims = []
    num_runs = 10
    for i in range(num_runs):
        start = time.time()
        area = computeArea()
        perim = computePerim()
        dt = time.time() - start

        areas.append(area)
        perims.append(perim)
        times.append(dt)

    err_area = np.array(areas) - actual_area

    avg_err_area = np.mean(err_area)
    std_err_area = np.std(err_area)

    err_perim = np.array(perims) - actual_perim

    avg_err_perim = np.mean(err_perim)
    std_err_perim = np.std(err_perim)

    prof.disable()
    print("Actual area is pi = {:.6f}, computed is {:.6f}, Average percent relative error is {:.4f}".format(actual_area,
                                                                                                            np.mean(areas), 100 * avg_err_area / actual_area))
    print("Actual perim is pi = {:.6f}, computed is {:.6f}, Average percent relative error is {:.4f}".format(actual_perim,
                                                                                                             np.mean(perims), 100 * avg_err_perim / actual_perim))
    print("Area error stats: mean = {}, stddev = {}".format(avg_err_area, std_err_area))
    print("Perim error stats: mean = {}, stddev = {}".format(avg_err_perim, std_err_perim))
    print("Took {:3} ms on average".format(np.mean(times) * 1000))

    prof.dump_stats('circ.stats')


if __name__ == "__main__":
    main()
