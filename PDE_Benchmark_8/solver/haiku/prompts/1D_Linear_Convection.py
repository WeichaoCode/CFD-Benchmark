import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
c = 1.0  # convection speed
epsilon = 5e-4  # damping factor
x_start, x_end = -5, 5
t_start, t_end = 0, 10

# Discretization parameters
nx = 200  # spatial points
nt = 1000  # time steps

# Grid generation
dx = (x_end - x_start) / (nx - 1)
x = np.linspace(x_start, x_end, nx)
dt = (t_end - t_start) / nt

# Initialize solution array 
u = np.exp(-x**2)

# Numerical solver (Lax-Wendroff scheme with periodic BC)
for n in range(nt):
    # Create copy of solution for update
    u_old = u.copy()
    
    # Periodic boundary handling
    u[0] = u_old[-2]
    u[-1] = u_old[1]
    
    # Lax-Wendroff scheme
    u[1:-1] = u_old[1:-1] - 0.5 * c * dt/dx * (u_old[2:] - u_old[:-2]) + \
              0.5 * epsilon * dt/dx**2 * (u_old[2:] - 2*u_old[1:-1] + u_old[:-2]) + \
              0.5 * c**2 * dt**2/dx**2 * (u_old[2:] - 2*u_old[1:-1] + u_old[:-2])

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/u_1D_Linear_Convection.npy', u)