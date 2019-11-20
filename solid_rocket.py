from constants import *
import numpy as np


def computeChamberPressure(burn_area):
	return (rho * a * c_star * burn_area / area_throat) ** (1. / (1. - n))

def computeExitVel():
	under_rad = gamma * R * computeExitTemp()

	assert under_rad >= 0, "Vel is less than zero!"
	return exit_mach * np.sqrt(under_rad)


def computeExitPressure(Pc):
	press = Pc * (1 + ((gamma - 1) / 2) * (exit_mach ** 2)) ** (- gamma / (gamma - 1))
	assert press >= 0, "Pressure is less than zero!"
	return press


def computeExitTemp():
	return T_internal * (1 + ((gamma - 1) / 2) * (exit_mach ** 2)) ** (-1)


def computeNozzleExitArea():
	# this power shows up twice
	powfac = (gamma + 1) / (2 * (gamma - 1))

	first = ((gamma + 1) / 2) ** (- powfac)
	second = (1 / exit_mach) * (1 + (exit_mach**2) * (gamma - 1) / 2) ** powfac

	return first * second
