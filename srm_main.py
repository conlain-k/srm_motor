import numpy as np
from matplotlib import pyplot as plt
import constants
from solid_rocket import *
from srm_system import *
import shapes

from cProfile import Profile
from pstats import Stats

from itertools import chain

fps = 25
steps_per_frame = 1 / (fps * constants.delta_t)

def burnSRM(srm, srm_type, make_plot=False):
    pressures = []
    thrusts = []
    areas = []
    x_vals = []

    t = 0
    step_num = 0

    regressed_dist = 0

    while(t < t_end and srm.isStillBurning()):
        A_cyl = srm.getArea()

        Pc = computeChamberPressure(A_cyl)
        r_dot = computeRdot(Pc)
        m_dot = computeMdot(r_dot, A_cyl)

        srm.advance(r_dot)

        Pexit = computeExitPressure(Pc)

        Vexit = computeExitVel()

        Fthrust = m_dot * computeExitVel() + (Pexit - Pambient) * computeNozzleExitArea()

        pressures.append(Pc)
        x_vals.append(constants.grain_length - srm.getLength())
        areas.append(A_cyl)
        thrusts.append(Fthrust)

        if (step_num % steps_per_frame == 0):
            print("t = {:.3f}: A = {}, F = {}, Pc = {}, rdot = {}".format(
                t, A_cyl, Fthrust, Pc, r_dot))
            if make_plot:
                figs = list(chain.from_iterable([srm.make_plot()]))
                for ind, f in enumerate(figs):
                    f.savefig("images/{}_{}_{}.png".format(srm_type,
                                                           ind, int(step_num // steps_per_frame)))
                    plt.close(f)
        t += constants.delta_t
        step_num += 1

    pressures = np.array(pressures)
    thrusts = np.array(thrusts)
    areas = np.array(areas)
    x_vals = np.array(x_vals)

    return pressures, thrusts, areas, x_vals


def runAndTimeSRM(srm, srm_type, make_plot=False):
    print("Testing {} SRM model!".format(srm_type))

    prof = Profile()

    prof.enable()
    P, T, A, x = burnSRM(srm, srm_type, make_plot)
    prof.disable()

    t = delta_t * np.arange(P.size)

    prof.dump_stats('{}_srm.stats'.format(srm_type))

    return P, T, A, x, t


def main():
    bates_srm = BatesSRM()

    shapecoll = shapes.ShapeCollection()
    circ = shapes.Circle((0, 0), R_inner)
    shapecoll.addShape(circ)

    bates_grain_ls = LevelSetSRM(shapecoll)

    theta = np.arange(0, 4) * np.pi / 4.
    lines_c = []

    lines = [shapes.LineSegment([finocyl_spoke_len * np.cos(t), finocyl_spoke_len * np.sin(
        t)], [-finocyl_spoke_len * np.cos(t), -finocyl_spoke_len * np.sin(t)], width=finocyl_spoke_width) for t in theta]
    [shapecoll.addShape(l) for l in lines]
    finocyl_grain_ls = LevelSetSRM(shapecoll)

    sustainer_assembly = SRM_Assembly({bates_grain_ls: 5})

    booster_assembly = SRM_Assembly({finocyl_grain_ls: 2, bates_grain_ls: 5})

    Pb, Tb, Ab, xb, tb = runAndTimeSRM(booster_assembly, "Booster", True)

    np.savetxt("results/Pb.txt", Pb)
    np.savetxt("results/Tb.txt", Tb)
    np.savetxt("results/Ab.txt", Ab)
    np.savetxt("results/xb.txt", xb)

    Psus, Tsus, Asus, xsus, tsus = runAndTimeSRM(sustainer_assembly, "Sustainer", True)
    np.savetxt("results/Psus.txt", Psus)
    np.savetxt("results/Tsus.txt", Tsus)
    np.savetxt("results/Asus.txt", Asus)
    np.savetxt("results/xsus.txt", xsus)


if __name__ == "__main__":
    main()
