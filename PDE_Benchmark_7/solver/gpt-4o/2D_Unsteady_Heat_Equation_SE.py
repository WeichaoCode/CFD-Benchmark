import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1  # Thermal diffusivity
Lx, Ly = 2.0, 2.0  # Domain size in x and y direction
nx, ny = 41, 41  # Number of grid points in x and y
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # Grid spacing
x = np.linspace(-1, 1, nx)  # Linearly spaced grid in x
y = np.linspace(-1, 1, ny)  # Linearly spaced grid in y
X, Y = np.meshgrid(x, y)  # 2D grid

t_max = 3.0  # Maximum simulation time
r = 0.2  # Stability parameter
dt = r * dx**2 / alpha  # Time step size
nt = int(t_max / dt) + 1  # Number of time steps

# Source term
Q0 = 200  # Source strength
sigma = 0.1  # Source width
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Initial temperature field
T = np.zeros((ny, nx))

# Main time-stepping loop
for n in range(nt):
    Tn = T.copy()
    
    # Update interior points using the explicit method
    T[1:-1, 1:-1] = (Tn[1:-1, 1:-1] + 
                     r * (Tn[2:, 1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1]) +
                     r * (Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) +
                     dt * q[1:-1, 1:-1])
    
    # Dirichlet boundary conditions (fixed temperature at boundaries)
    T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0

# Save the final temperature distribution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/T_2D_Unsteady_Heat_Equation_SE.npy', T)

# Visualization of the final temperature distribution
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Final Temperature Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()