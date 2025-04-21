#!/usr/bin/env python3
import numpy as np

# Parameters
Re_tau = 395.0
mu = 1.0 / Re_tau          # Molecular viscosity
kappa = 0.42               # von Kármán constant
A = 25.4                   # Damping constant

# Domain discretization
y_start = 0.0
y_end = 2.0
N = 201                   # number of grid points
y = np.linspace(y_start, y_end, N)
dy = y[1] - y[0]

# Compute effective viscosity (Cess turbulence model)
# Compute y+ = y*Re_tau
y_plus = y * Re_tau

# Compute the term X in the Cess model
# Note: (2y - y^2)^2 and (3 - 4y + 2y^2)^2 appear in the formula.
term1 = (2*y - y**2)**2
term2 = (3 - 4*y + 2*y**2)**2
# Compute the damping function: [1 - exp(-y_plus/A)]^2
damping = (1.0 - np.exp(-y_plus / A))**2
X = (1.0/9.0) * kappa**2 * Re_tau**2 * term1 * term2 * damping

# Cess model: mu_eff/mu = 1/2*(sqrt(1+X)-1)
ratio = 0.5 * (np.sqrt(1.0 + X) - 1.0)
mu_eff = mu * ratio

# For the momentum equation, effective viscosity is taken to be mu_eff = mu + mu_t.
# Thus, we solve for u given d/dy(mu_eff du/dy) = -1.
# Discretization: central differences with Dirichlet BCs: u(0)=0 and u(2)=0.
# Build coefficients for interior nodes i = 1 to N-2.
A_diag = np.zeros(N-2)
B_diag = np.zeros(N-2)
C_diag = np.zeros(N-2)
RHS = -np.ones(N-2)  # Right-hand side, constant -1

# Compute mu_eff at half-nodes by averaging adjacent nodes.
mu_half = np.zeros(N-1)
for i in range(N-1):
    mu_half[i] = 0.5 * (mu_eff[i] + mu_eff[i+1])

# Assembly of the linear system for interior nodes
for i in range(0, N-2):
    if i == 0:
        a = mu_half[i] / (dy**2)         # coefficient for u[0] (boundary) -- u[0]=0 so not stored in matrix
    else:
        a = mu_half[i] / (dy**2)
    b = - (mu_half[i] + mu_half[i+1]) / (dy**2)
    if i == N-3:
        c = mu_half[i+1] / (dy**2)         # coefficient for u[N-1] (boundary) -- u[N-1]=0 so not stored in matrix
    else:
        c = mu_half[i+1] / (dy**2)
    A_diag[i] = a
    B_diag[i] = b
    C_diag[i] = c

# Build the tridiagonal matrix
# The system corresponds to interior nodes u[1]...u[N-2]
diag = B_diag.copy()
lower = A_diag[1:]   # from i=1 to N-3
upper = C_diag[:-1]  # from i=0 to N-3

# Assemble full matrix
n_interior = N - 2
LHS = np.zeros((n_interior, n_interior))
for i in range(n_interior):
    LHS[i, i] = diag[i]
    if i > 0:
        LHS[i, i-1] = lower[i-1]
    if i < n_interior - 1:
        LHS[i, i+1] = upper[i]

# Solve linear system
u_interior = np.linalg.solve(LHS, RHS)

# Construct full solution vector u with BCs
u = np.zeros(N)
u[1:N-1] = u_interior
u[0] = 0.0
u[-1] = 0.0

# Compute mu_t from mu_eff; since mu_eff = mu + mu_t
mu_t = mu_eff - mu

# Other initial conditions (constant fields)
k = np.full(N, 0.01)
epsilon = np.full(N, 0.001)
omega = np.full(N, 1.0)
nu_SA = np.full(N, mu)  # nu_SA = 1/Re_tau = mu

# Save the final solution fields as .npy files (1D arrays)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/mu_t_Fully_Developed_Turbulent_Channel_Flow.npy', mu_t)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/k_Fully_Developed_Turbulent_Channel_Flow.npy', k)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/epsilon_Fully_Developed_Turbulent_Channel_Flow.npy', epsilon)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/omega_Fully_Developed_Turbulent_Channel_Flow.npy', omega)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/nu_SA_Fully_Developed_Turbulent_Channel_Flow.npy', nu_SA)