import numpy as np

# Parameters
nu = 0.3          # Diffusion coefficient
L = 2.0           # Length of domain
T_final = 0.0333  # Final time

# Discretization parameters
dx = 0.01
x = np.arange(0, L + dx, dx)
Nx = len(x)
# Stability condition for explicit scheme: dt <= dx^2 / (2*nu)
dt = 0.0001
Nt = int(T_final / dt)

# Initial condition
u = np.ones(Nx)
u[(x >= 0.5) & (x <= 1.0)] = 2.0

# Time stepping loop for the unsteady diffusion problem
for n in range(Nt):
    u_new = u.copy()
    # Apply Dirichlet boundary conditions (assumed same as initial, u=1 at boundaries)
    u_new[0] = 1.0
    u_new[-1] = 1.0
    # Update internal nodes using explicit finite-difference method
    u_new[1:-1] = u[1:-1] + nu * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
    u = u_new

# Save final solution as the variable 'u' in a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_1D_Diffusion.npy', u)