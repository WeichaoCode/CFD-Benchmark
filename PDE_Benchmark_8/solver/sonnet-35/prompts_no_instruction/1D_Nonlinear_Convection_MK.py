import numpy as np
import math

# Problem parameters
L = 2 * np.pi  # Domain length
T = 500  # Number of time steps
dt = 0.01  # Time step
nu = 0.5  # Numerical parameter
dx = dt / nu  # Spatial step

# Grid setup
x = np.linspace(0, L, math.ceil(L / (dt / nu)))
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# MacCormack method for solving the advection equation
for _ in range(T):
    # Predictor step
    u_pred = np.zeros_like(u)
    for i in range(nx):
        left_index = (i-1+nx) % nx
        u_pred[i] = u[i] - dt/dx * u[i] * (u[i] - u[left_index])
    
    # Corrector step
    u_new = np.zeros_like(u)
    for i in range(nx):
        left_index = (i-1+nx) % nx
        right_index = (i+1) % nx
        
        flux_pred = u_pred[i] * (u_pred[i] - u_pred[right_index])
        flux_curr = u[i] * (u[i] - u[left_index])
        
        u_new[i] = u[i] - 0.5 * dt/dx * (flux_pred + flux_curr)
    
    u = u_new

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_1D_Nonlinear_Convection_MK.npy', u)