import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

# Parameters
Nr = 100  # Number of radial points
m = 1     # Azimuthal wavenumber
kz = 1    # Axial wavenumber
Re = 1e4  # Reynolds number

# Grid
r = np.linspace(0.001, 1, Nr)  # Avoid r=0 singularity
dr = r[1] - r[0]

# Background flow
w0 = 1 - r**2
dw0dr = -2*r

# Differential operators in r
# First derivative
D1 = sparse.diags([-1, 1], [-1, 1], shape=(Nr, Nr))/(2*dr)
D1 = D1.tolil()
D1[0, 0:3] = [-3/(2*dr), 2/dr, -1/(2*dr)]  # One-sided at r=0
D1[-1, -3:] = [1/(2*dr), -2/dr, 3/(2*dr)]  # One-sided at r=1

# Second derivative 
D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(Nr, Nr))/dr**2
D2 = D2.tolil()
D2[0, 0:4] = [2/dr**2, -5/dr**2, 4/dr**2, -1/dr**2]  # At r=0
D2[-1, -3:] = [1/dr**2, -2/dr**2, 1/dr**2]  # At r=1

# Build system matrix
I = sparse.eye(Nr)
Z = sparse.csr_matrix((Nr, Nr))

# Operators for vector Laplacian
L_u = D2 - I/r**2 - m**2*sparse.diags(1/r**2) - kz**2*I
L_v = L_u - 2*m*sparse.diags(1/(r**2))
L_w = L_u

# Build block matrix for eigenvalue problem
A11 = -L_u/Re
A12 = -2*m*sparse.diags(1/(r**2))/Re
A13 = -kz*D1
A21 = 2*m*sparse.diags(1/(r**2))/Re
A22 = -L_v/Re
A23 = -1j*m*sparse.diags(1/r)
A31 = kz*D1
A32 = 1j*m*sparse.diags(1/r)
A33 = -L_w/Re

B11 = I + kz*sparse.diags(w0)
B12 = Z
B13 = Z
B21 = Z 
B22 = I + kz*sparse.diags(w0)
B23 = Z
B31 = sparse.diags(dw0dr)
B32 = Z
B33 = I + kz*sparse.diags(w0)

# Assemble full matrices in LIL format first
A = sparse.bmat([[A11, A12, A13],
                 [A21, A22, A23],
                 [A31, A32, A33]], format='lil')

B = sparse.bmat([[B11, B12, B13],
                 [B21, B22, B23],
                 [B31, B32, B33]], format='lil')

# Apply boundary conditions while still in LIL format
for i in range(3):
    idx = i*Nr - 1
    if idx >= 0:
        A[idx,:] = 0
        A[idx,idx] = 1
        B[idx,:] = 0
        B[idx,idx] = 1

# Convert to CSC format for eigenvalue solver
A = A.tocsc()
B = B.tocsc()

# Solve eigenvalue problem
eigenvalues, eigenvectors = eigs(A, k=1, M=B, sigma=0.5+0.1j, which='LM')

# Extract most unstable mode
u = eigenvectors[:Nr,0]
v = eigenvectors[Nr:2*Nr,0] 
w = eigenvectors[2*Nr:,0]

# Normalize
norm = np.max(np.abs(w))
u = u/norm
v = v/norm 
w = w/norm

# Save results
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_Pipe_Flow_Disk_EVP.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/v_Pipe_Flow_Disk_EVP.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/w_Pipe_Flow_Disk_EVP.npy', w)