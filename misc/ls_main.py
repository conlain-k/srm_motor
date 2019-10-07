from srm_system import *
from level_set import *
import constants

import matplotlib
import scipy
import scipy.ndimage

# matplotlib.use('agg')

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors

X = []
Y = []
Vel = 0.2

def showframe(SDF, title_str, max_v):
    fig, ax = plt.subplots()
    
    inside_vals = np.where(SDF <= 0)
    BC_vals = np.where(X**2 + Y**2 > constants.R_outer ** 2)

    SDF_i_bc = np.copy(SDF)
    SDF_i_bc[BC_vals] = np.nan
    SDF_mbc = midpoints(SDF_i_bc)

    SDF_i_bc[inside_vals] = np.nan
    SDF_mid = midpoints(SDF_i_bc)

    sdfplot = ax.pcolormesh(X, Y,
                            np.ma.masked_where(np.isnan(SDF_mbc), SDF_mid), cmap='coolwarm_r', vmax=constants.R_outer)

    SDF_mid[:] = 0

    cMap = colors.ListedColormap(['slategray'])

    ax.pcolormesh(X, Y,
                  np.ma.masked_where(np.invert(np.isnan(SDF_mbc)), SDF_mid), cmap=cMap, vmax=constants.R_outer)

    # sdfplot = ax.imshow(midpoints(SDF))
    # ax.contour(X, Y, SDF, colors='black', levels=10)
    ax.set_title("Level set solver {}".format(title_str))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = fig.colorbar(sdfplot, ax=ax)
    cb.set_label("Distance from burn front")
    return fig

def animate(SDF_vals, frame_idx):
    print("animating frame {}".format(frame_idx))
    SDF_i = np.copy(SDF_vals[frame_idx])
    vol = computeInteriorVolume(SDF_i)
    area = computeSurfaceArea(SDF_i)
    print("volume is {}".format(vol))
    print("Area is {}".format(area))
    return showframe(SDF_i, ", t = {}, Vol. = {:3f}, $A_s$ = {:3f}".format(frame_idx / fps, vol, area), np.nanmax(SDF_vals[0]))


def main():
    shapes = ShapeCollection()

    finocyl_spoke_width = 0.05 * constants.R_outer
    
    line1 = LineSegment([-1, 0], [1, 0], width=finocyl_spoke_width)
    line2 = LineSegment([0, -1], [0, 1], width=finocyl_spoke_width)
    # shapes.addShape(line1)
    # shapes.addShape(line2)

    circ = Circle([0, 0], constants.R_inner)
    shapes.addShape(circ)

    ls_motor = LevelSetSRM(shapes)

    SDF_init = ls_motor.getSDF()

    global X
    global Y
    X, Y = ls_motor.getGrid()

    print("final volume is {}".format(np.pi * constants.R_outer ** 2))

    fig, ax = plt.subplots(2, 2)
    im00 = ax[0, 0].imshow(SDF_init)
    fig.colorbar(im00, ax=ax[0, 0])
    im01 = ax[0, 1].imshow(computeNorm(computeGrad(SDF_init)))
    fig.colorbar(im01, ax=ax[0, 1])

    im10 = ax[1, 0].imshow(computeUpwindGrad(SDF_init))
    fig.colorbar(im10, ax=ax[1, 0])
    im11 = ax[1, 1].imshow(computeNorm(computeGrad(SDF_init)) - computeUpwindGrad(SDF_init))
    fig.colorbar(im11, ax=ax[1, 1])
    plt.show()

    # start with initial value
    SDF_vals = [ls_motor.getSDF()]

    sdfplot = animate(SDF_vals, 0)
    plt.show()

    for step_num in np.arange(1, num_steps + 1):
        # grad = computeGrad(SDF)

        ls_motor.integrate(Vel)

        plt.show()

        if (step_num % steps_per_frame == 0):
            print("t = {}".format(step_num * constants.delta_t))
            SDF_vals.append(ls_motor.getSDF())

    sdfplot = animate(SDF_vals, -1)

    plt.show()

    for frame_idx in range(len(SDF_vals)):
        fig = animate(SDF_vals, frame_idx)
        plt.savefig("images/level_set_{}.png".format(int(frame_idx)), dpi=150)
        plt.close()

    plt.show()


if __name__ == "__main__":
    main()