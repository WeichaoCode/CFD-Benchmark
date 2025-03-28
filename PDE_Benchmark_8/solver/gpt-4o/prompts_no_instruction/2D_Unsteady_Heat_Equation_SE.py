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

# Create grid
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
T = np.zeros((ny, nx))

# Time-stepping loop
t = 0.0
while t < t_max:
    T_new = np.copy(T)
    
    # Update interior points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            d2T_dx2 = (T[j, i+1] - 2*T[j, i] + T[j, i-1]) / dx**2
            d2T_dy2 = (T[j+1, i] - 2*T[j, i] + T[j-1, i]) / dy**2
            q = Q0 * np.exp(-(x[i]**2 + y[j]**2) / (2 * sigma**2))
            T_new[j, i] = T[j, i] + dt * (alpha * (d2T_dx2 + d2T_dy2) + q)
    
    # Apply Dirichlet boundary conditions
    T_new[0, :] = 0.0
    T_new[-1, :] = 0.0
    T_new[:, 0] = 0.0
    T_new[:, -1] = 0.0
    
    # Update time and temperature field
    T = T_new
    t += dt

# Save the final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/T_2D_Unsteady_Heat_Equation_SE.npy', T)