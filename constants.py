import numpy as np

# Compute thrust and pressure curves for cylindrical SRB
# R_inner = 1.2255 / 2.
R_inner = 1 / 2.
R_outer = 3.225 / 2.
grain_length = 5.346405
finocyl_spoke_width = 0.05 * R_outer
finocyl_spoke_len = 0.7 * R_outer


diam_throat = 1.05
area_throat = np.pi * (diam_throat / 2) ** 2

# Burn rate coeff (in / s) (lbf / in^2) ^ -n
a = 0.027
# pressure exp
n = 0.3
# specific heat ratio
gamma = 1.2
# exit vel
exit_mach = 2.
# specific gas constant
R = 2000.
# exit temperature
T_internal = 4800.
# characteristic burn vel (feet per second)
c_star = 4890.
# propellant density (slug / in^3)
rho = 0.00176

# timestep
delta_t = 5e-4

# cutoff end time
t_end = 10

#  pressure (lbf / in^2)
Pambient = 14.7
