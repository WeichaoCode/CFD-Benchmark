import numpy as np
import matplotlib.pyplot as plt

# Define Parameters
c = 1.0  # Convection speed
epsilon_undamped = 0.0  # Damping factor for undamped case
epsilon_damped = 5e-4  # Damping factor for damped case

L = 10  # Length of domain from -5 to 5
Nx = 101  # Number of spatial grid points
dx = L / (Nx - 1)  # Spatial step size

# CFL condition
sigma = 0.5  # Courant number
dt = sigma * dx / abs(c)

# Discretize the spatial domain
x = np.linspace(-5, 5, Nx)

# Initial condition
u0 = np.exp(-x**2)

# Function to compute spatial derivatives using central difference
def compute_spatial_derivatives(u, dx):
    # Derivatives using central difference with periodic boundary conditions
    dudx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    d2udx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    return dudx, d2udx2 

# Function to advance solution using Predictor-Corrector method
def advance_time_step(u, dt, dx, c, epsilon):
    dudx, d2udx2 = compute_spatial_derivatives(u, dx)
    f_n = -c * dudx + epsilon * d2udx2
    
    # Predictor step
    u_star = u + dt * f_n
    
    # Recalculate derivatives at u_star
    dudx_star, d2udx2_star = compute_spatial_derivatives(u_star, dx)
    f_star = -c * dudx_star + epsilon * d2udx2_star
    
    # Corrector step
    u_next = u + (dt / 2) * (f_n + f_star)
    
    return u_next

# Function to run simulation
def run_simulation(epsilon, title, timesteps=200):
    u = u0.copy()
    results = [u.copy()]  # Store initial condition
    
    for _ in range(timesteps):
        u = advance_time_step(u, dt, dx, c, epsilon)
        # Apply periodic boundary conditions
        u[0] = u[-1]
        u[-1] = u[0]
        results.append(u.copy())
    
    # Save results to a .npy file
    np.save(f'wave_{title.lower().replace(" ", "_")}.npy', np.array(results))
    
    return results

# Run simulations
undamped_results = run_simulation(epsilon_undamped, "Undamped", timesteps=100)
damped_results = run_simulation(epsilon_damped, "Damped", timesteps=100)

# Visualization
def plot_results(results, title):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    
    times = [0, 25, 50, 75, 100]  # Time steps to plot
    for i in times:
        plt.plot(x, results[i], label=f't={i*dt:.2f}')
    
    plt.legend()
    plt.show()

# Plot results
plot_results(undamped_results, "Undamped Wave Propagation")
plot_results(damped_results, "Damped Wave Propagation")