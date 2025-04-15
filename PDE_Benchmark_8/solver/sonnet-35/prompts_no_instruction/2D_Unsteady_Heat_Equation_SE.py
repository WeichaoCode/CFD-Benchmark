import numpy as np

# Problem parameters
nx, ny = 41, 41
Lx, Ly = 2.0, 2.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
alpha = 1.0  # Thermal diffusivity
Q0 = 200.0   # Source intensity
sigma = 0.1  # Source width
r = 0.25     # Stability coefficient
dt = r * dx**2 / alpha
t_max = 3.0

# Grid generation
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
T = 1 + Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Boundary conditions
T[0, :] = 1
T[-1, :] = 1
T[:, 0] = 1
T[:, -1] = 1

# Time integration
for t in np.arange(dt, t_max + dt, dt):
    # Create temporary array for update
    T_new = T.copy()
    
    # Interior points update
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Source term
            q = Q0 * np.exp(-(X[i,j]**2 + Y[i,j]**2) / (2 * sigma**2))
            
            # Explicit finite difference
            T_new[i,j] = T[i,j] + alpha * dt * (
                (T[i+1,j] - 2*T[i,j] + T[i-1,j]) / dx**2 + 
                (T[i,j+1] - 2*T[i,j] + T[i,j-1]) / dy**2
            ) + dt * q

    # Update solution
    T = T_new

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/T_2D_Unsteady_Heat_Equation_SE.npy', T)