import numpy as np

# Parameters
c = 1.0  # Convection speed
epsilon = 5e-4  # Damping factor
x_start, x_end = -5, 5  # Spatial domain
N_x = 101  # Number of spatial grid points
dx = (x_end - x_start) / (N_x - 1)  # Spatial step size
x = np.linspace(x_start, x_end, N_x)  # Spatial grid

# Initial condition
u_initial = np.exp(-x**2)

# CFL condition for stability
CFL = 0.5
dt = CFL * dx / c  # Time step size

# Time integration parameters
t_final = 2.0  # Final time
N_t = int(t_final / dt)  # Number of time steps

# Function to compute spatial derivatives
def compute_derivatives(u):
    # Periodic boundary conditions
    u_x = np.zeros_like(u)
    u_xx = np.zeros_like(u)
    
    # Central difference for first derivative
    u_x[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    u_x[0] = (u[1] - u[-1]) / (2 * dx)
    u_x[-1] = (u[0] - u[-2]) / (2 * dx)
    
    # Central difference for second derivative
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2)
    u_xx[0] = (u[1] - 2 * u[0] + u[-1]) / (dx**2)
    u_xx[-1] = (u[0] - 2 * u[-1] + u[-2]) / (dx**2)
    
    return u_x, u_xx

# Function to compute the right-hand side of the PDE
def rhs(u):
    u_x, u_xx = compute_derivatives(u)
    return -c * u_x + epsilon * u_xx

# 4th-order Runge-Kutta method
def rk4_step(u, dt):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Time integration loop
u = u_initial.copy()
for _ in range(N_t):
    u = rk4_step(u, dt)

# Save the final solution to a .npy file
np.save('final_solution.npy', u)