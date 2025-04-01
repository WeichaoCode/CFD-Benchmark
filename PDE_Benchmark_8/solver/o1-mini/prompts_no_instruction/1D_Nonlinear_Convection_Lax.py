import numpy as np
import math

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
N = math.ceil(L / dx)
x = np.linspace(0, L, N, endpoint=False)
u = np.sin(x) + 0.5 * np.sin(0.5 * x)
T = 500

# Lax method time integration
for _ in range(T):
    u_plus = np.roll(u, -1)
    u_minus = np.roll(u, 1)
    f_plus = 0.5 * u_plus**2
    f_minus = 0.5 * u_minus**2
    u = 0.5 * (u_plus + u_minus) - (dt / (2 * dx)) * (f_plus - f_minus)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/u_1D_Nonlinear_Convection_Lax.npy', u)