import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Nx = 101  # Number of spatial grid points
x_start, x_end = -5.0, 5.0  # Spatial domain
c = 1.0  # Convection speed
epsilon_values = [0.0, 5e-4]  # Damping factors (undamped and damped cases)
dx = (x_end - x_start) / (Nx - 1)  # Spatial step size
x = np.linspace(x_start, x_end, Nx)  # Spatial grid

# Time-related parameters
CFL = 0.5  # CFL condition for stability
dt = CFL * dx / c  # Time step
T = 2.0  # Total time
Nt = int(T / dt)  # Number of time steps

# Initial condition
def initial_condition(x):
    return np.exp(-x**2)

# Central difference for spatial derivatives
def central_diff(u, epsilon):
    # First derivative (convection term)
    du_dx = np.roll(u, -1) - np.roll(u, 1)  # Periodic boundary conditions
    du_dx = c * du_dx / (2 * dx)

    # Second derivative (diffusion term)
    d2u_dx2 = np.roll(u, -1) - 2 * u + np.roll(u, 1)  # Periodic boundary conditions
    d2u_dx2 = epsilon * d2u_dx2 / (dx**2)

    return -du_dx + d2u_dx2  # minus sign due to convention

# Runge-Kutta 4th order advancement
def rk4_step(u, dt, epsilon):
    k1 = central_diff(u, epsilon)
    k2 = central_diff(u + 0.5 * dt * k1, epsilon)
    k3 = central_diff(u + 0.5 * dt * k2, epsilon)
    k4 = central_diff(u + dt * k3, epsilon)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Simulation main loop
def run_simulation(epsilon):
    u = initial_condition(x)
    u_record = [u.copy()]

    for n in range(Nt):
        u = rk4_step(u, dt, epsilon)
        u_record.append(u.copy())

    return np.array(u_record)

# Visualization and running simulations for different epsilon values
for epsilon in epsilon_values:
    results = run_simulation(epsilon)
    label = 'Damped' if epsilon > 0 else 'Undamped'
    
    # Save the final time step profile
    np.save(f"wave_profile_{label}.npy", results[-1, :])
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x, results[0, :], label='Initial')
    plt.plot(x, results[int(Nt/2), :], label='Mid-time')
    plt.plot(x, results[-1, :], label='Final')
    plt.title(f"Wave Propagation - {label} case")
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    plt.show()