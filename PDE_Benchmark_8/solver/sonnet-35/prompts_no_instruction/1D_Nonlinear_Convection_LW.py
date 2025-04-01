import numpy as np
import math

# Problem parameters
L = 2 * np.pi  # Domain length
T = 500  # Number of time steps
dt = 0.01  # Time step
nu = 0.5  # Numerical parameter
dx = dt / nu  # Spatial step size
nx = math.ceil(L / dx)  # Number of spatial points

# Initialize grid
x = np.linspace(0, L, nx)
u = np.sin(x) + 0.5 * np.sin(0.5 * x)  # Initial condition

# Lax-Wendroff Method
for _ in range(T):
    # Compute fluxes
    f = 0.5 * u**2
    
    # Flux derivatives
    df_dx_plus = np.roll(f, -1) - f
    df_dx_minus = f - np.roll(f, 1)
    
    # Lax-Wendroff update
    u_plus = 0.5 * (np.roll(u, -1) + np.roll(u, 1)) - \
             0.5 * dt/dx * (df_dx_plus + df_dx_minus) + \
             0.5 * (dt/dx)**2 * (df_dx_plus * df_dx_minus)
    
    # Periodic boundary condition
    u_plus[-1] = u_plus[0]
    u = u_plus

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_1D_Nonlinear_Convection_LW.npy', u)