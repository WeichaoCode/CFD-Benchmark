#!/usr/bin/env python3
import numpy as np

# Parameters
alpha = 0.01
Q0 = 200.0
sigma = 0.1

# Domain
x_start, x_end = -1.0, 1.0
y_start, y_end = -1.0, 1.0
t_final = 3.0

Nx = 101
Ny = 101
x = np.linspace(x_start, x_end, Nx)
y = np.linspace(y_start, y_end, Ny)
dx = (x_end - x_start) / (Nx - 1)
dy = (y_end - y_start) / (Ny - 1)

# Time stepping parameters (stability condition for explicit diffusion: dt <= min(dx,dy)**2/(4*alpha))
dt = 0.005
Nt = int(t_final/dt)

# Create mesh
X, Y = np.meshgrid(x, y, indexing='ij')

# Initial condition: T(x,y,0) = 1 + 200 * exp(-((x^2+y^2) / (2*0.1^2)))
T = 1.0 + 200.0 * np.exp(- (X**2 + Y**2) / (2.0 * sigma**2))

# Define source term: q(x,y,t) = Q0 * exp(-((x^2+y^2) / (2*sigma^2)))
q = Q0 * np.exp(- (X**2 + Y**2) / (2.0 * sigma**2))

# Enforce boundary conditions at t=0
T[0, :] = 1.0
T[-1, :] = 1.0
T[:, 0] = 1.0
T[:, -1] = 1.0

# Time integration: explicit finite difference method
for n in range(Nt):
    T_new = T.copy()
    # Update for interior nodes
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * (
        alpha * (
            (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[0:-2, 1:-1])/(dx**2) +
            (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, 0:-2])/(dy**2)
        ) + q[1:-1, 1:-1]
    )
    # Enforce Dirichlet boundary conditions
    T_new[0, :] = 1.0
    T_new[-1, :] = 1.0
    T_new[:, 0] = 1.0
    T_new[:, -1] = 1.0
    T = T_new

# Save the final solution array as T.npy (2D numpy array)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/T_2D_Unsteady_Heat_Equation.npy', T)