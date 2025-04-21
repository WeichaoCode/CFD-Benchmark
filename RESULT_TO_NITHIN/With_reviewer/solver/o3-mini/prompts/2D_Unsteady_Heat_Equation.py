#!/usr/bin/env python3
import numpy as np

# Parameters
alpha = 0.01   # thermal diffusivity
Q0 = 200.0     # source magnitude in °C/s
sigma = 0.1    # source standard deviation

# Domain definition
x_start, x_end = -1.0, 1.0
y_start, y_end = -1.0, 1.0
t_start, t_end = 0.0, 3.0

# Grid resolution
Nx = 101
Ny = 101
dx = (x_end - x_start) / (Nx - 1)
dy = (y_end - y_start) / (Ny - 1)
x = np.linspace(x_start, x_end, Nx)
y = np.linspace(y_start, y_end, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Time step (satisfying explicit scheme stability condition)
dt = 0.005
nsteps = int((t_end - t_start) / dt)

# Precompute the source term q(x,y)
q = Q0 * np.exp(-((X**2 + Y**2) / (2 * sigma**2)))

# Initial condition: T(x,y,0) = 1 + 200*exp(-((x^2+y^2)/(2*0.1^2)))
T = 1.0 + 200.0 * np.exp(-((X**2 + Y**2) / (2 * sigma**2)))

# Enforce boundary conditions: T(x,y,t) = 1 on all boundaries
T[0, :] = 1.0
T[-1, :] = 1.0
T[:, 0] = 1.0
T[:, -1] = 1.0

# Time integration (Explicit Forward Euler)
for step in range(nsteps):
    Tn = T.copy()
    
    # Compute Laplacian using central difference for interior points
    laplacian = (
        (Tn[2:, 1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1]) / dx**2 +
        (Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) / dy**2
    )
    
    # Update interior points using the PDE: T_t = alpha * Laplacian + q
    T[1:-1, 1:-1] = Tn[1:-1, 1:-1] + dt * (alpha * laplacian + q[1:-1, 1:-1])
    
    # Reapply boundary condition
    T[0, :] = 1.0
    T[-1, :] = 1.0
    T[:, 0] = 1.0
    T[:, -1] = 1.0

# Save the final solution T as a 2D NumPy array in a file named T.npy
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/T_2D_Unsteady_Heat_Equation.npy', T)