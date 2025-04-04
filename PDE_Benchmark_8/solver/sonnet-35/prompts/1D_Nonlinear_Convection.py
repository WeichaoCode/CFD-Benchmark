import numpy as np

# Problem parameters
L = 2 * np.pi  # Domain length
T = 5.0        # Total simulation time
Nx = 200       # Number of spatial points 
Nt = 500       # Number of time points

# Grid generation
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]

# CFL condition and time step
CFL = 0.5
dt = CFL * dx

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Computational solution using Lax-Friedrichs method
for _ in range(Nt):
    # Create copy of current solution
    u_old = u.copy()
    
    # Compute numerical flux using Lax-Friedrichs splitting
    for i in range(1, Nx-1):
        # Centered difference approximation with dissipation
        u[i] = 0.5 * (u_old[i+1] + u_old[i-1]) - \
               0.5 * dt/dx * (u_old[i+1] - u_old[i-1])
    
    # Enforce periodic boundary conditions
    u[0] = u[-2]
    u[-1] = u[1]

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/u_1D_Nonlinear_Convection.npy', u)