#!/usr/bin/env python3
import numpy as np

# Parameters
n = 3.0
R0 = 5.0
N = 101         # Total grid points (including r=1 boundary)
r_start = 0.0
r_end = 1.0
max_iter = 50
tol = 1e-8
dr = (r_end - r_start) / (N - 1)

# Create radial grid
r = np.linspace(r_start, r_end, N)

# Initial guess: f0(r)= R0^(2/(n-1))*(1 - r^2)^2. For n=3, exponent=1.
f = R0 * (1 - r**2)**2
f[-1] = 0.0  # enforce Dirichlet BC at r=1

# Number of unknowns (all except the last grid point where f=0 is prescribed)
M = N - 1

def residual_and_jacobian(f):
    R = np.zeros(M)
    J = np.zeros((M, M))
    # i = 0, at r=0: use the regularity condition
    R[0] = 2*(f[1] - f[0]) / dr**2 + f[0]**n
    J[0, 0] = -2/dr**2 + n * f[0]**(n-1)
    J[0, 1] = 2/dr**2
    
    # i = 1 to M-2 (interior unknowns)
    for i in range(1, M - 1):
        R[i] = (f[i+1] - 2*f[i] + f[i-1]) / dr**2 + (f[i+1] - f[i-1])/(dr * r[i]) + f[i]**n
        J[i, i-1] = 1/dr**2 - 1/(dr * r[i])
        J[i, i]   = -2/dr**2 + n * f[i]**(n-1)
        J[i, i+1] = 1/dr**2 + 1/(dr * r[i])
    
    # i = M-1: last unknown with BC f(N)=0
    i = M - 1
    R[i] = (0.0 - 2*f[i] + f[i-1]) / dr**2 + (0.0 - f[i-1])/(dr * r[i]) + f[i]**n
    J[i, i-1] = 1/dr**2 - 1/(dr * r[i])
    J[i, i]   = -2/dr**2 + n * f[i]**(n-1)
    return R, J

# Newton-Raphson iteration with fixed damping to avoid large steps
damping = 0.5
for iteration in range(max_iter):
    R, J = residual_and_jacobian(f)
    normR = np.linalg.norm(R, ord=2)
    if normR < tol:
        break
    try:
        delta = np.linalg.solve(J, -R)
    except np.linalg.LinAlgError:
        break
    f[:M] += damping * delta
    # Optional: clip extreme values to avoid numerical overflow
    f[:M] = np.clip(f[:M], -1e6, 1e6)

# Save final solution f as a 1D numpy array in 'f.npy'
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/f_Lane_Emden_Equation.npy', f)