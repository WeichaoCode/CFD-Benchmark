import numpy as np

# Problem parameters
Lx, Ly = 1.0, 1.0  # Domain size
nx, ny = 100, 100  # Number of grid points
nt = 300  # Number of time steps
alpha = 1.0  # Thermal diffusivity 
Q0 = 200.0  # Source magnitude
sigma = 0.1  # Source width

# Grid generation
dx = 2*Lx / (nx-1)
dy = 2*Ly / (ny-1)
dt = 3.0 / nt

x = np.linspace(-Lx, Lx, nx)
y = np.linspace(-Ly, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
T = np.ones((ny, nx), dtype=np.float64) + Q0 * np.exp(-(X**2 + Y**2)/(2*sigma**2))

# Stability check
stability = alpha * dt / min(dx**2, dy**2)
print(f"Stability check (should be <= 0.5): {stability}")

# Time stepping (explicit finite difference)
for n in range(nt):
    # Source term
    q = Q0 * np.exp(-(X**2 + Y**2)/(2*sigma**2))
    
    # Create copy of previous time step
    T_old = T.copy()
    
    # Compute Laplacian
    laplacian_x = np.zeros_like(T_old[1:-1, 1:-1], dtype=np.float64)
    laplacian_y = np.zeros_like(T_old[1:-1, 1:-1], dtype=np.float64)
    
    # Compute second-order central differences
    laplacian_x = (T_old[1:-1, 2:] - 2*T_old[1:-1, 1:-1] + T_old[1:-1, :-2]) / (dx**2)
    laplacian_y = (T_old[2:, 1:-1] - 2*T_old[1:-1, 1:-1] + T_old[:-2, 1:-1]) / (dy**2)
    
    # Update temperature
    T[1:-1, 1:-1] = (T_old[1:-1, 1:-1] + 
                     alpha * dt * (laplacian_x + laplacian_y) + 
                     dt * q[1:-1, 1:-1])
    
    # Boundary conditions
    T[0, :] = 1.0
    T[-1, :] = 1.0
    T[:, 0] = 1.0
    T[:, -1] = 1.0

# Save final solution
np.save('/PDE_Benchmark/results/prediction/sonnet-35/prompts/T_2D_Unsteady_Heat_Equation.npy', T)