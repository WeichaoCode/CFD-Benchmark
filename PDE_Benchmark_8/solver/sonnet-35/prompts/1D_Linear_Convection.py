import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
c = 1.0  # convection speed
epsilon = 5e-4  # damping factor
x_start, x_end = -5, 5
t_start, t_end = 0, 10

# Discretization parameters
nx = 200  # spatial points
nt = 500  # time steps

# Grid setup
dx = (x_end - x_start) / (nx - 1)
dt = (t_end - t_start) / nt
x = np.linspace(x_start, x_end, nx)

# Initial condition
u = np.exp(-x**2)

# Numerical scheme (Lax-Wendroff method)
def lax_wendroff_step(u, c, epsilon, dx, dt):
    # Compute fluxes and diffusion terms
    u_plus = np.roll(u, -1)
    u_minus = np.roll(u, 1)
    
    # Lax-Wendroff flux terms
    flux_plus = 0.5 * c * (u + u_plus) - 0.5 * c**2 * dt/dx * (u_plus - u)
    flux_minus = 0.5 * c * (u_minus + u) - 0.5 * c**2 * dt/dx * (u - u_minus)
    
    # Diffusion term
    diff_term = epsilon * (u_plus - 2*u + u_minus) / dx**2
    
    # Update solution
    u_new = u - dt/dx * (flux_plus - flux_minus) + dt * diff_term
    
    # Periodic boundary conditions
    u_new[0] = u_new[-1]
    
    return u_new

# Time integration
for _ in range(nt):
    u = lax_wendroff_step(u, c, epsilon, dx, dt)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/u_1D_Linear_Convection.npy', u)