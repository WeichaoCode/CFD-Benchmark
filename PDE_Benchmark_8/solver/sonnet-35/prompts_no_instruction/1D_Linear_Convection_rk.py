import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
Lx = 10  # Domain length
Nx = 101  # Number of spatial points
x_start, x_end = -5, 5  # Domain boundaries
epsilon = 5e-4  # Damping factor
c = 1.0  # Convection speed

# Grid setup
x = np.linspace(x_start, x_end, Nx)
dx = x[1] - x[0]

# Time parameters
t_end = 2.0
CFL = 0.1
dt = CFL * dx / np.abs(c)
Nt = int(t_end / dt)

# Initial condition
u = np.exp(-x**2)

# RK4 right-hand side function
def rhs(u):
    # Periodic boundary conditions via circular shift
    u_periodic = np.zeros_like(u)
    u_periodic[:-1] = u[1:]
    u_periodic[-1] = u[0]
    
    # 2nd order central difference for spatial derivatives
    du_dx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    d2u_dx2 = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (dx**2)
    
    return -c * du_dx + epsilon * d2u_dx2

# RK4 time integration
for _ in range(Nt):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    
    u += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# Save solution
save_values = ['u']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_1D_Linear_Convection_rk.npy', u)