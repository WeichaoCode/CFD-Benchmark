import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0           # Thermal diffusivity
Q0 = 200.0            # Heat source amplitude (°C/s)
sigma = 0.1           # Source width
nx, ny = 41, 41       # Number of grid points
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_max = 3.0           # Maximum simulation time (s)

# Grid setup
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
beta = dx / dy
r = 0.25  # Stability parameter (must satisfy (1 + beta^2)*r <= 0.5)

# Time step
dt = r * dx**2 / alpha
n_steps = int(t_max / dt)

# Initialize temperature field
T = np.zeros((ny, nx))

# Create meshgrid for source term
X, Y = np.meshgrid(x, y)
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Time-stepping loop
for step in range(n_steps):
    T_new = T.copy()
    # Update interior points
    T_new[1:-1, 1:-1] = (
        T[1:-1, 1:-1]
        + r * (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1])
        + (beta**2) * r * (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2])
        + dt * q[1:-1, 1:-1]
    )
    # Apply Dirichlet boundary conditions (0°C)
    T_new[0, :] = 0
    T_new[-1, :] = 0
    T_new[:, 0] = 0
    T_new[:, -1] = 0
    T = T_new

# Visualization
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, T, 50, cmap='hot')
plt.colorbar(cp)
plt.title('Temperature Distribution at t = {:.2f} s'.format(t_max))
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()

# Save the final temperature distribution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/T_2D_Unsteady_Heat_Equation_SE.npy', T)