import numpy as np
import math

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
T = 500

# Discretize the spatial domain
x = np.linspace(0, L, math.ceil(L / dx))
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# MacCormack method
for n in range(T):
    # Predictor step
    u_pred = np.copy(u)
    for i in range(nx - 1):
        u_pred[i] = u[i] - (dt / dx) * u[i] * (u[i+1] - u[i])
    
    # Apply periodic boundary condition for predictor
    u_pred[-1] = u[-1] - (dt / dx) * u[-1] * (u[0] - u[-1])
    
    # Corrector step
    u_corr = np.copy(u_pred)
    for i in range(1, nx):
        u_corr[i] = 0.5 * (u[i] + u_pred[i] - (dt / dx) * u_pred[i] * (u_pred[i] - u_pred[i-1]))
    
    # Apply periodic boundary condition for corrector
    u_corr[0] = 0.5 * (u[0] + u_pred[0] - (dt / dx) * u_pred[0] * (u_pred[0] - u_pred[-1]))
    
    # Update solution
    u = np.copy(u_corr)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/u_1D_Nonlinear_Convection_MK.npy', u)