import numpy as np

# Parameters
nu = 0.3
L = 2.0
T_final = 0.0333
nx = 201
dx = L / (nx - 1)

# CFL condition for stability: dt <= dx^2 / (2*nu)
dt = 0.0001  # a conservative choice (dt <= dx^2/(2*nu))
nt = int(T_final / dt)

# Spatial grid
x = np.linspace(0, L, nx)

# Initial condition: u(x,0) = 2 if 0.5 <= x <= 1, else 1
u = np.ones(nx)
u[(x >= 0.5) & (x <= 1.0)] = 2.0

# Time integration using the FTCS scheme for the diffusion equation
for n in range(nt):
    u_new = u.copy()
    # Apply interior point update
    for i in range(1, nx-1):
        u_new[i] = u[i] + nu * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    # Boundary conditions - Dirichlet: values are fixed at the initial condition (1 on boundaries)
    u_new[0] = 1.0
    u_new[-1] = 1.0
    u = u_new

# Save final solution as a 1D numpy array in a .npy file with the variable name "u"
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_1D_Diffusion.npy', u)