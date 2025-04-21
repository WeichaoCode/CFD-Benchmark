#!/usr/bin/env python3
import numpy as np

# Parameters
Re_tau = 395.0
mu = 1.0 / Re_tau      # Molecular viscosity
kappa = 0.42           # von Kármán constant
A = 25.4               # Damping constant

# Domain and discretization
y_start = 0.0
y_end = 2.0
N = 201                # number of grid points
y = np.linspace(y_start, y_end, N)
dy = y[1] - y[0]

# Compute effective viscosity from Cess turbulence model at grid points
y_plus = y * Re_tau
Q = (1.0/9.0) * kappa**2 * Re_tau**2 * (2*y - y**2)**2 * (3 - 4*y + 2*y**2)**2 * (1 - np.exp(-y_plus/A))**2
mu_eff = mu * (0.5 * (np.sqrt(1 + Q) - 1.0))

# Solve the steady momentum equation:
# d/dy( mu_eff * du/dy ) = -1 with u(0)=0, u(2)=0
# Discretize: (1/dy^2) * [ mu_eff_{i+1/2}*(u[i+1]-u[i]) - mu_eff_{i-1/2}*(u[i]-u[i-1]) ] = -1

# Number of unknowns (excluding boundary nodes)
N_internal = N - 2

# Assemble coefficient matrix A and right-hand side vector b for internal nodes
A_matrix = np.zeros((N_internal, N_internal))
b = -np.ones(N_internal)  # right-hand side is -1 for each interior node

# Use central differences with harmonic averaging for mu_eff at interfaces = average here
for i in range(N_internal):
    # global index corresponding to interior node
    I = i + 1
    # compute mu_eff at interfaces by arithmetic average
    mu_plus = 0.5 * (mu_eff[I] + mu_eff[I+1])
    mu_minus = 0.5 * (mu_eff[I] + mu_eff[I-1])
    # diagonal
    A_matrix[i, i] = (mu_minus + mu_plus) / (dy**2)
    # lower diagonal
    if i - 1 >= 0:
        A_matrix[i, i-1] = - mu_minus / (dy**2)
    # upper diagonal
    if i + 1 < N_internal:
        A_matrix[i, i+1] = - mu_plus / (dy**2)

# Solve the linear system
u_internal = np.linalg.solve(A_matrix, b)

# Assemble full solution vector u including boundary conditions u(0)=0 and u(end)=0.
u = np.zeros(N)
u[1:-1] = u_internal

# Compute mu_t = mu_eff - mu (eddy viscosity)
mu_t = mu_eff - mu

# Define other variables as constant initial conditions over the domain
k = np.full_like(y, 0.01)
epsilon = np.full_like(y, 0.001)
omega = np.full_like(y, 1.0)
nu_SA = np.full_like(y, 1.0/Re_tau)

# Save final solution variables as 1D numpy arrays in .npy files.
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/mu_t_Fully_Developed_Turbulent_Channel_Flow.npy', mu_t)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/k_Fully_Developed_Turbulent_Channel_Flow.npy', k)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/epsilon_Fully_Developed_Turbulent_Channel_Flow.npy', epsilon)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/omega_Fully_Developed_Turbulent_Channel_Flow.npy', omega)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/nu_SA_Fully_Developed_Turbulent_Channel_Flow.npy', nu_SA)