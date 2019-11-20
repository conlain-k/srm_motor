import numpy as np
from matplotlib import pyplot as plt
import constants

def computeTotalImpulse(thrusts):
    return np.trapz(thrusts, dx = constants.delta_t)


def createPlots():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("F (lbf)")
    axes[0].set_title("Thrust $vs.$ time")
    axes[0].set_adjustable("box")
    axes[0].grid()

    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("$P_c$ (lbf / in$^2$)")
    axes[1].set_title("Pressure $vs.$ time")
    axes[1].set_adjustable("box")
    axes[1].grid()

    axes[2].set_xlabel("x (in)")
    axes[2].set_ylabel("$A_s$ (in$^2$)")
    axes[2].set_title("Area $vs.$ regression distance")
    axes[2].set_adjustable("box")
    axes[2].grid()

    fig.suptitle("SRM burn results")

    return fig, axes


def makeLegends(fig, axes):
    axes[0].legend(loc = 'upper right')
    axes[1].legend(loc = 'upper right')
    axes[2].legend(loc = 'upper right')


def plotResults(fig, axes, pressures, thrusts, areas, x_vals, times, srm_type, symbol):
    axes[0].plot(times, thrusts, symbol, label=srm_type)
    axes[1].plot(times, pressures, symbol, label=srm_type)
    axes[2].plot(x_vals, areas, symbol, label=srm_type)

def main():
    Psus = np.loadtxt("results/Psus.txt")
    Tsus = np.loadtxt("results/Tsus.txt")
    Asus = np.loadtxt("results/Asus.txt")
    xsus = np.loadtxt("results/xsus.txt")

    tsus = constants.delta_t * np.arange(0, Psus.size)

    Pb = np.loadtxt("results/Pb.txt")
    Tb = np.loadtxt("results/Tb.txt")
    Ab = np.loadtxt("results/Ab.txt")
    xb = np.loadtxt("results/xb.txt")

    tb = constants.delta_t * np.arange(0, Pb.size)

    Pbates = np.loadtxt("results/Pbates.txt")
    Tbates = np.loadtxt("results/Tbates.txt")
    Abates = np.loadtxt("results/Abates.txt")
    xbates = np.loadtxt("results/xbates.txt")

    tbates = constants.delta_t * np.arange(0, Pbates.size)

    imp_bates_analyt = computeTotalImpulse(Tbates)
    imp_bates = computeTotalImpulse(Tb)
    imp_sus = computeTotalImpulse(Tsus)

    fig, axes = createPlots()

    plotResults(fig, axes, Pb, Tb, Ab, xb, tb, "Booster \nImpulse = {:.3f}".format(imp_bates), '-')
    plotResults(fig, axes, Psus, Tsus, Asus, xsus, tsus, "Sustainer \nImpulse = {:.3f}".format(imp_sus), '--')
    plotResults(fig, axes, Pbates, Tbates, Abates, xbates, tbates, "Sustainer (analytical) \nImpulse = {:.3f}".format(imp_bates_analyt), ':')
    makeLegends(fig, axes)
    plt.savefig("srm_results.png")
    plt.show()

if __name__ == '__main__':
    main()

