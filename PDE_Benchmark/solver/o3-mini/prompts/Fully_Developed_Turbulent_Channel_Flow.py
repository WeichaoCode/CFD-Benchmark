#!/usr/bin/env python3
import numpy as np

# Parameters
Re_tau = 395.0
mu = 1.0 / Re_tau         # Molecular viscosity
rho = 1.0                 # Fluid density (not explicitly used)
L = 2.0                   # Domain length in y
N = 201                   # Number of grid points
dy = L / (N - 1)
y = np.linspace(0, L, N)

# Turbulent eddy viscosity model (sample profile)
# This is a simple model that peaks in the center and goes to zero at the walls.
mu_t = 0.1 * y * (L - y)

# Effective viscosity as function of y
a = mu + mu_t  # a(y) = mu + mu_t

# Assemble the linear system Au = b for interior points (Dirichlet BC at y=0 and y=L)
# Using central finite differences for variable coefficient second derivative:
# (1/dy^2)[a(i+1/2)*(u[i+1]-u[i]) - a(i-1/2)*(u[i]-u[i-1])] = -1.
N_int = N - 2  # Number of interior points
A = np.zeros((N_int, N_int))
b = -np.ones(N_int)  # Right-hand side -1

# Construct the coefficients for interior points i=1 to N-2 (index i_int = i-1)
for i in range(1, N-1):
    i_int = i - 1

    # Compute a at half points using arithmetic average
    a_plus = 0.5 * (a[i] + a[i+1])      # a_{i+1/2}
    a_minus = 0.5 * (a[i] + a[i-1])       # a_{i-1/2}

    # Diagonal coefficient
    A[i_int, i_int] = (a_minus + a_plus) / (dy**2)
    
    # Lower off-diagonal (if not at the first interior)
    if i_int - 1 >= 0:
        A[i_int, i_int - 1] = -a_minus / (dy**2)
    
    # Upper off-diagonal (if not at the last interior)
    if i_int + 1 < N_int:
        A[i_int, i_int + 1] = -a_plus / (dy**2)

# Adjust the RHS for Dirichlet BC: u(0) = 0 and u(L)=0 (but they are zero, so no adjustments needed)

# Solve the linear system for interior velocities
u_int = np.linalg.solve(A, b)

# Construct the full solution array and set boundary conditions
u = np.zeros(N)
u[1:-1] = u_int

# Save the final solution as a 1D numpy array named 'u.npy'
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)

# Also save the coordinate array y in case it is needed for postprocessing (optional)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/y_Fully_Developed_Turbulent_Channel_Flow.npy', y)