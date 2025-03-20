import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 41, 41  # Grid points
x, y = np.linspace(-1, 1, nx), np.linspace(-1, 1, ny)
dx, dy = x[1] - x[0], y[1] - y[0]  # Grid spacing
dt, t_max = 0.001, 3.0  # Time parameters
alpha = 1.0  # Thermal diffusivity
sigma, Q0 = 0.1, 200.0  # Source term parameters
r = alpha * dt / dx**2  # Adjusted stability parameter
beta_sq = (dx / dy) ** 2

# Stability parameter modification as per the provided relationship
r = r / (1 + beta_sq)

# Initialize temperature field
T = np.zeros((nx, ny))
T_new = np.zeros_like(T)
T_old = np.zeros_like(T)

# Source term
X, Y = np.meshgrid(x, y, indexing='ij')
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Time-stepping loop
time = 0.0
while time < t_max:
    # Apply DuFort-Frankel update
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            T_new[i, j] = ((2 * r * (T[i + 1, j] + T[i - 1, j]) +
                           2 * beta_sq * r * (T[i, j + 1] + T[i, j - 1]) +
                           T_old[i, j] + 2 * dt * q[i, j]) /
                           (1 + 2 * r + 2 * beta_sq * r))
    
    # Update old temperatures
    T_old[:, :] = T[:, :]
    T[:, :] = T_new[:, :]
    
    # Increment time
    time += dt

# Save the final temperature distribution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/T_2D_Unsteady_Heat_Equation_DF.npy', T)

# Visualization
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, T, cmap='hot', levels=50)
plt.colorbar(label='Temperature (°C)')
plt.title("Temperature Distribution at t = {:.2f}s".format(time))
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()