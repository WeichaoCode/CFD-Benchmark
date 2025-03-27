import numpy as np

# Parameters
alpha = 0.01  # Thermal diffusivity
Q0 = 200.0  # Source term strength
sigma = 0.1  # Source term spread
nx, ny = 41, 41  # Grid resolution
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_max = 3.0  # Maximum time
r = 0.25  # Stability factor

# Derived parameters
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
dt = r * dx**2 / alpha
nt = int(t_max / dt)  # Number of time steps

# Create grid
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
T = np.zeros((ny, nx))

# Time-stepping loop
for n in range(nt):
    Tn = T.copy()
    # Update the temperature field
    T[1:-1, 1:-1] = (Tn[1:-1, 1:-1] +
                     alpha * dt / dx**2 * (Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) +
                     alpha * dt / dy**2 * (Tn[2:, 1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1]) +
                     dt * Q0 * np.exp(-((X[1:-1, 1:-1]**2 + Y[1:-1, 1:-1]**2) / (2 * sigma**2))))

    # Apply Dirichlet boundary conditions
    T[:, 0] = 0
    T[:, -1] = 0
    T[0, :] = 0
    T[-1, :] = 0

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/T_2D_Unsteady_Heat_Equation_SE.npy', T)