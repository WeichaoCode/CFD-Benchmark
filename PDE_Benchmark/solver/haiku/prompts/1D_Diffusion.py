import numpy as np

# Problem parameters
nu = 0.3  # Diffusion coefficient
L = 2.0    # Domain length 
T = 0.0333 # Total simulation time

# Discretization parameters
nx = 200   # Number of spatial points
nt = 500   # Number of time steps

# Grid generation
dx = L / (nx - 1)
dt = T / (nt - 1)
x = np.linspace(0, L, nx)

# Initialize solution array
u = np.ones_like(x)
u[(x >= 0.5) & (x <= 1.0)] = 2.0

# Stability check (CFL condition)
alpha = nu * dt / (dx**2)
print(f"Stability parameter (alpha): {alpha}")
if alpha > 0.5:
    raise ValueError("Unstable numerical scheme. Reduce dt or increase dx.")

# Time-stepping (Explicit Finite Difference Method)
for _ in range(nt - 1):
    u_old = u.copy()
    
    # Interior points (exclude boundaries)
    u[1:-1] = u_old[1:-1] + nu * dt / (dx**2) * (
        u_old[2:] - 2 * u_old[1:-1] + u_old[:-2]
    )
    
    # Neumann boundary conditions (zero flux)
    u[0] = u[1]
    u[-1] = u[-2]

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_1D_Diffusion.npy', u)