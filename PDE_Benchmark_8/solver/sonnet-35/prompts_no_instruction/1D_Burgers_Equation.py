import numpy as np

# Problem parameters
nu = 0.07
nx = 101
nt = 100
L = 2 * np.pi
dx = L / (nx - 1)
dt = dx * nu

# Grid setup
x = np.linspace(0, L, nx)

# Initial condition function
def phi(x):
    return np.exp(-x**2 / (4*nu)) + np.exp(-(x - L)**2 / (4*nu))

# Initial condition
u = -2 * nu / phi(x) * np.gradient(phi(x), dx) + 4

# Numerical scheme with explicit stabilization
for _ in range(nt):
    u_old = u.copy()
    
    # Compute spatial derivatives
    du_dx = np.gradient(u_old, dx)
    d2u_dx2 = np.gradient(du_dx, dx)
    
    # Stabilized central difference scheme
    u[1:-1] = u_old[1:-1] - 0.5 * dt * (
        u_old[2:] * du_dx[2:] - u_old[:-2] * du_dx[:-2]
    ) / dx + nu * dt * d2u_dx2[1:-1] / (dx**2)
    
    # Periodic boundary conditions
    u[0] = u[-2]
    u[-1] = u[1]

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_1D_Burgers_Equation.npy', u)