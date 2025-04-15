import numpy as np
import math

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
N = math.ceil(L / dx)
x = np.linspace(0, L, N, endpoint=False)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Time stepping
T = 500
for _ in range(T):
    f = 0.5 * u**2
    f_plus = np.roll(f, -1)
    f_minus = np.roll(f, 1)
    u = u - (dt / (2 * dx)) * (f_plus - f_minus) + (dt**2 / (2 * dx**2)) * u * (f_plus - 2 * f + f_minus)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/u_1D_Nonlinear_Convection_LW.npy', u)