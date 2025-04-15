import numpy as np
import matplotlib.pyplot as plt

# Define parameters
x_start, x_end = -5.0, 5.0
Nx = 101
dx = (x_end - x_start) / (Nx - 1)
c = 1.0  # Convection speed
epsilon_undamped = 0.0
epsilon_damped = 5e-4
T = 2.0  # Simulation time

# CFL condition for time step
CFL = 0.5
dt = CFL * dx / c

# Discretize the domain
x = np.linspace(x_start, x_end, Nx)

# Initial condition
def initial_condition(x):
    return np.exp(-x**2)

# Explicit Euler Method for time integration
def solve_convection_diffusion(epsilon, Nt):
    # Initialize u
    u = initial_condition(x)
    u_new = np.copy(u)
    
    # Time integration
    for n in range(Nt):
        # Central difference for first derivative
        dudx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        
        # Central difference for second derivative
        d2udx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx**2)
        
        # Update using explicit Euler
        u_new = u - c * dt * dudx + epsilon * dt * d2udx2
        
        # Periodic boundary conditions
        u_new[0] = u_new[-2]
        u_new[-1] = u_new[1]
        
        # Update u
        u[:] = u_new
    
    return u

# Calculate the number of time steps
Nt = int(T / dt)

# Solve for undamped case
u_undamped = solve_convection_diffusion(epsilon_undamped, Nt)

# Solve for damped case
u_damped = solve_convection_diffusion(epsilon_damped, Nt)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x, initial_condition(x), label='Initial')
plt.plot(x, u_undamped, label='Undamped ($\\epsilon=0$)')
plt.plot(x, u_damped, label='Damped ($\\epsilon=5\\times10^{-4}$)')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Wave Profile after $t = 2$')
plt.legend()
plt.grid()
plt.savefig('convection_diffusion_plot.png')
plt.show()

# Save results to .npy files
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/u_1D_Linear_Convection_explicit_euler.npy', u_damped)