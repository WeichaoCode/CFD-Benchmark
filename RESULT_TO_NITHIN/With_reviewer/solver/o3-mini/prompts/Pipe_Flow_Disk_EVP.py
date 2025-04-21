#!/usr/bin/env python3
import numpy as np
from scipy.linalg import eig

# Parameters and domain
N = 100                # number of grid points in r
r_start = 0.0
r_end = 1.0
Re = 1e4
kz = 1.0               # axial wavenumber (real)
dr = (r_end - r_start) / (N - 1)
r = np.linspace(r_start, r_end, N)

# Helper: finite difference derivative matrices (second order accuracy)
D = np.zeros((N, N), dtype=np.complex128)
D2 = np.zeros((N, N), dtype=np.complex128)

# First derivative matrix D
for i in range(1, N-1):
    D[i, i-1] = -1.0/(2*dr)
    D[i, i+1] =  1.0/(2*dr)
# Forward difference at i=0
D[0, 0] = -1.0/dr
D[0, 1] =  1.0/dr
# Backward difference at i=N-1
D[N-1, N-2] = -1.0/dr
D[N-1, N-1] =  1.0/dr

# Second derivative matrix D2
for i in range(1, N-1):
    D2[i, i-1] = 1.0/(dr**2)
    D2[i, i]   = -2.0/(dr**2)
    D2[i, i+1] = 1.0/(dr**2)
# For boundaries use second order one-sided differences
D2[0,0] = 2.0/(dr**2)
D2[0,1] = -5.0/(dr**2)
D2[0,2] = 4.0/(dr**2)
D2[0,3] = -1.0/(dr**2)
D2[N-1, N-3] = -1.0/(dr**2)
D2[N-1, N-2] = 4.0/(dr**2)
D2[N-1, N-1] = -5.0/(dr**2)
D2[N-1, N-4] = 2.0/(dr**2)

# Diagonal matrices
I_N = np.eye(N, dtype=np.complex128)
# For 1/r: at r=0, use limit value 0 (regularity)
r_inv = np.zeros(N, dtype=np.complex128)
r_inv[1:] = 1.0 / r[1:]
R_inv = np.diag(r_inv)
R_diag = np.diag(r)

# Background flow w0 = 1 - r^2 and its radial derivative
w0 = 1.0 - r**2
dw0_dr = -2.0 * r
W0 = np.diag(w0)
dW0 = np.diag(dw0_dr)

# Construct the composite radial Laplacian operator L = d2/dr2 + (1/r)*d/dr - kz^2*I.
L = D2 + R_inv @ D - (kz**2)*I_N

# Construct the operator for the divergence in continuity: (1/r)d/dr(r*•)
# Define operator C = diag(1/r) * (diag(r)*D). At r=0, set value via limit (use D[0] row already computed).
C = np.zeros((N, N), dtype=np.complex128)
# For i>=1 use formula; for i=0, we approximate using the forward difference result:
for i in range(N):
    # Multiply D acting on (r*variable): first form diag(r)*D then multiply by r_inv.
    # We build matrix: C = R_inv*(R_diag @ D)
    pass
C = R_inv @ (R_diag @ D)  # note: at r=0, r_inv[0]=0 so that row becomes 0

# Total number of unknowns: u (radial velocity), w (axial velocity), p (pressure)
# Ordering: x = [u (0:N); w (N:2N); p (2N:3N)]
n_total = 3 * N

# Initialize global matrices A and B for the generalized eigenvalue problem: A x = s B x.
A = np.zeros((n_total, n_total), dtype=np.complex128)
B = np.zeros((n_total, n_total), dtype=np.complex128)

# ----------------------------------------------------------------
# Block 1: Continuity equation: (1/r d/dr(r u)) + i*kz*w = 0, no eigenvalue s.
# Rows 0 ... N-1
A[0:N, 0:N] = C
A[0:N, N:2*N] = 1j * kz * I_N
# p does not appear in continuity equation, so A[0:N, 2*N:3*N] remains 0.
# B block remains 0.
# ----------------------------------------------------------------
# Block 2: Radial momentum: s*u + i*kz*w0*u + d(p)/dr - (1/Re)*L u = 0.
# Rows N ... 2N-1
rows2 = slice(N, 2*N)
# s*u term: will be in B. So for u-component, B block gets identity.
B[rows2, 0:N] = I_N
# The remaining (s-independent) part in A:
A[rows2, 0:N] = 1j * kz * W0 - (1.0/Re)*L
# Add d/dr acting on p: use derivative matrix D for p (acting on p variable).
A[rows2, 2*N:3*N] = D
# No w terms in radial momentum (set zero)
# ----------------------------------------------------------------
# Block 3: Axial momentum: s*w + i*kz*w0*w + (dw0/dr)*u + i*kz*p - (1/Re)*L w = 0.
# Rows 2N ... 3N-1
rows3 = slice(2*N, 3*N)
# s*w term: in B, w block gets identity.
B[rows3, N:2*N] = I_N
# u coupling term:
A[rows3, 0:N] = dW0  # diag(dw0/dr)
# w terms:
A[rows3, N:2*N] = 1j * kz * W0 - (1.0/Re)*L
# p term:
A[rows3, 2*N:3*N] = 1j * kz * I_N
# ----------------------------------------------------------------
# Impose Boundary Conditions (BCs)
# We replace selected rows in A and B with BC rows.

# 1. u no-slip at r=0: u(0) = 0.
# u is solved in radial momentum (Block 2). For i=0, row index = N (first row of block2)
row_bc_u0 = N  # corresponding to u at r=0 in radial momentum block
A[row_bc_u0, :] = 0.0
B[row_bc_u0, :] = 0.0
# Set u(0)=1
A[row_bc_u0, 0] = 1.0

# 2. u no-slip at r=1: u(1) = 0.
# For r=1, i = N-1 in u, row index in block2 = N + (N-1) = 2*N - 1.
row_bc_u1 = 2*N - 1
A[row_bc_u1, :] = 0.0
B[row_bc_u1, :] = 0.0
# Set u(N-1)=1; note u component indices 0:N, so position N-1.
A[row_bc_u1, N-1] = 1.0

# 3. w no-slip at r=1: w(1) = 0.
# w is in axial momentum block. For r=1, i = N-1 in w, row index = 2*N + (N-1) = 3*N - 1.
row_bc_w1 = 3*N - 1
A[row_bc_w1, :] = 0.0
B[row_bc_w1, :] = 0.0
# Set w(N-1)=1; w block is indices N:2*N, so index = N + (N-1) = 2*N - 1.
A[row_bc_w1, 2*N - 1] = 1.0

# 4. Pressure gauge: set p(0) = 0.
# p is in the third block (columns 2N:3N). We impose this in the axial momentum block.
# Use row corresponding to p at r=0 in Block 3: row index = 2*N (first row of block3)
row_bc_p0 = 2*N
A[row_bc_p0, :] = 0.0
B[row_bc_p0, :] = 0.0
# p(0) is at column index 2*N (first entry in p block).
A[row_bc_p0, 2*N] = 1.0

# ----------------------------------------------------------------
# Solve the generalized eigenvalue problem A x = s B x.
# Note: Many rows in B are zero; the eigenvalue solver will give spurious infinite eigenvalues.
# We select eigenvalues with finite magnitude and choose the one with maximum real part.
eigvals, eigvecs = eig(A, B)

# Filter out eigenvalues with extremely large magnitude (spurious due to BC enforcement)
finite_mask = np.abs(eigvals) < 1e6
eigvals = eigvals[finite_mask]
eigvecs = eigvecs[:, finite_mask]

# Choose the eigenvalue with maximum real part
index = np.argmax(eigvals.real)
s_eig = eigvals[index]
x_eig = eigvecs[:, index]

# Normalize eigenvector for convenience
x_eig = x_eig / np.max(np.abs(x_eig))

# Extract variables: u, w, p as 1D arrays along r (size N each)
u_sol = x_eig[0:N].real
w_sol = x_eig[N:2*N].real
p_sol = x_eig[2*N:3*N].real

# Save final solutions in .npy files
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/u_sol_Pipe_Flow_Disk_EVP.npy', u_sol)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/w_sol_Pipe_Flow_Disk_EVP.npy', w_sol)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/p_sol_Pipe_Flow_Disk_EVP.npy', p_sol)