import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.01  # thermal diffusivity
Q0 = 200.0  # source term strength
sigma = 0.1  # source term spread
nx, ny = 41, 41  # grid resolution
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_max = 3.0  # maximum time
r = 0.25  # stability factor

# Grid setup
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Time step
dt = r * dx**2 / alpha
nt = int(t_max / dt)  # number of time steps

# Initialize temperature field
T = np.zeros((ny, nx))

# Source term
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Time-stepping loop
for n in range(nt):
    Tn = T.copy()
    T[1:-1, 1:-1] = (Tn[1:-1, 1:-1] +
                     alpha * dt / dx**2 * (Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) +
                     alpha * dt / dy**2 * (Tn[2:, 1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1]) +
                     dt * q[1:-1, 1:-1])

    # Apply Dirichlet boundary conditions
    T[:, 0] = 0
    T[:, -1] = 0
    T[0, :] = 0
    T[-1, :] = 0

# Save the final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/T_2D_Unsteady_Heat_Equation_SE.npy', T)

# Visualization
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Field at t = {:.2f} s'.format(t_max))
plt.xlabel('x')
plt.ylabel('y')
plt.show()