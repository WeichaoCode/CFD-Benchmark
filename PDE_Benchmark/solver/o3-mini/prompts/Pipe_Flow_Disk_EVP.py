#!/usr/bin/env python3
import numpy as np
import scipy.linalg

# Parameters
N = 50               # number of radial grid points
Re = 1e4
kz = 1.0
h = 1.0/(N-1)
r = np.linspace(0, 1, N)
w0 = 1 - r**2       # base flow
dw0_dr = -2*r       # derivative of base flow

# Total number of unknowns for variables u, w, p (each is an array of length N)
nVar = 3 * N

# Total number of equations:
# We'll use the PDE at interior points: i=1,...,N-2 for each of 3 equations (divergence, radial mom, axial mom) -> 3*(N-2)
# plus 6 boundary conditions:
#    u(0)=0, u(1)=0, w(1)=0, w'(0)=0, p(0)=0, p(1)=0.
nEq = 3*(N-2) + 6

# Create A and B matrices for the generalized eigenvalue problem A x = s B x.
A = np.zeros((nEq, nVar), dtype=complex)
B = np.zeros((nEq, nVar), dtype=complex)

# Helper indices for variable blocks
def ind_u(i):
    return i
def ind_w(i):
    return N + i
def ind_p(i):
    return 2*N + i

# Interior equations for i = 1,...,N-2
# Row counters:
# eq1: divergence equation
# eq2: radial momentum equation
# eq3: axial momentum equation

row = 0
for i in range(1, N-1):
    # ---------- Equation 1: Divergence (incompressibility) ---------------
    # (1/r) d/dr (r u) + i*kz*w = 0  at r_i.
    # Use centered finite difference for d/dr (r*u):
    #   (r[i+1]*u[i+1] - r[i-1]*u[i-1])/(2h) divided by r[i]
    if r[i] != 0:
        A[row, ind_u(i+1)] =  r[i+1] / (2*h*r[i])
        A[row, ind_u(i-1)] = -r[i-1] / (2*h*r[i])
    else:
        # At r=0, use symmetry; but i never equals 0 here.
        pass
    # Add term for w:
    A[row, ind_w(i)] = 1j * kz
    # (No s terms)
    # B row remains zero.
    row += 1

for i in range(1, N-1):
    # ---------- Equation 2: Radial momentum ---------------
    # Eq2: s * u + i*w0*u + dp/dr - (1/Re)*( u_rr + (1/r) u_r - (1/r^2+kz^2) u ) = 0, at r_i.
    # s multiplies u; so B: coefficient 1 on u_i.
    Arow = row
    B[Arow, ind_u(i)] = 1.0
    # Add advection term: i*w0[i]* u[i]
    A[Arow, ind_u(i)] += 1j * w0[i]
    # Finite difference for u_rr and u_r at interior point i.
    # Second derivative u_rr approximated by central differences:
    #   u[i-1]: 1/h^2, u[i]: -2/h^2, u[i+1]: 1/h^2.
    # First derivative u_r approximated by central differences:
    #   u[i-1]: -1/(2*h), u[i+1]: 1/(2*h).
    # Multiply first derivative by (1/r[i]).
    coef_u_im1 = - (1/Re)* ( (1.0/h**2) - (1/(2*h*r[i])) )
    coef_u_i   = - (1/Re)* ( (-2.0/h**2) - ( (1/r[i]**2 + kz**2) ) )
    coef_u_ip1 = - (1/Re)* ( (1.0/h**2) + (1/(2*h*r[i])) )
    A[Arow, ind_u(i-1)] += coef_u_im1
    A[Arow, ind_u(i)]   += coef_u_i
    A[Arow, ind_u(i+1)] += coef_u_ip1
    # Pressure gradient dp/dr approximated by central difference:
    #   p[i+1]: 1/(2*h), p[i-1]: -1/(2*h)
    A[Arow, ind_p(i+1)] +=  1.0/(2*h)
    A[Arow, ind_p(i-1)] += -1.0/(2*h)
    row += 1

for i in range(1, N-1):
    # ---------- Equation 3: Axial momentum ---------------
    # Eq3: s*w + i*w0*w + (dw0/dr)*u + i*kz*p - (1/Re)*(w_rr + (1/r)w_r - kz**2*w) = 0
    # s multiplies w; so B: coefficient 1 on w[i].
    Arow = row
    B[Arow, ind_w(i)] = 1.0
    # Advection term in w:
    A[Arow, ind_w(i)] += 1j * w0[i]
    # Viscous term for w: second derivative and first derivative
    coef_w_im1 = - (1/Re)* ( (1.0/h**2) - (1/(2*h*r[i])) )
    coef_w_i   = - (1/Re)* ( (-2.0/h**2) - kz**2 )
    coef_w_ip1 = - (1/Re)* ( (1.0/h**2) + (1/(2*h*r[i])) )
    A[Arow, ind_w(i-1)] += coef_w_im1
    A[Arow, ind_w(i)]   += coef_w_i
    A[Arow, ind_w(i+1)] += coef_w_ip1
    # Coupling from u through d(w0)/dr = -2r
    A[Arow, ind_u(i)] += dw0_dr[i]
    # Pressure term: i*kz*p
    A[Arow, ind_p(i)] += 1j * kz
    row += 1

# Boundary conditions (set in the remaining 6 rows)
# There are 6 BC equations. Their rows go from row to nEq-1.
# idx_BC_start = 3*(N-2)
idx_BC = row
# BC1: u(0) = 0
A[idx_BC, ind_u(0)] = 1.0
idx_BC += 1
# BC2: u(1) [i.e. at r=1, index N-1] = 0
A[idx_BC, ind_u(N-1)] = 1.0
idx_BC += 1
# BC3: w(1) = w at r=1 (index N-1) = 0
A[idx_BC, ind_w(N-1)] = 1.0
idx_BC += 1
# BC4: w'(0) = 0, use forward difference: (w(1)-w(0))/h = 0
A[idx_BC, ind_w(0)] = -1.0/h
A[idx_BC, ind_w(1)] = 1.0/h
idx_BC += 1
# BC5: p(0)=0
A[idx_BC, ind_p(0)] = 1.0
idx_BC += 1
# BC6: p(1)=0, i.e., at r=1 (index N-1)
A[idx_BC, ind_p(N-1)] = 1.0
idx_BC += 1

# Solve generalized eigenvalue problem A x = s B x.
# Use scipy.linalg.eig on full matrices.
eigvals, eigvecs = scipy.linalg.eig(A, B)

# Choose the eigenmode with maximum real part of s
max_index = np.argmax(eigvals.real)
s_eig = eigvals[max_index]
X = eigvecs[:, max_index]

# Extract solution fields: u, w, p are stored in X, ordering: first N -> u, next N -> w, last N -> p.
u_sol = X[0:N]
w_sol = X[N:2*N]
p_sol = X[2*N:3*N]

# Save final solution fields as .npy files (1D arrays)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_sol_Pipe_Flow_Disk_EVP.npy', u_sol)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/w_sol_Pipe_Flow_Disk_EVP.npy', w_sol)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_sol_Pipe_Flow_Disk_EVP.npy', p_sol)

print("Computed eigenvalue s = {}".format(s_eig))
print("Solution saved in u.npy, w.npy, and p.npy.")