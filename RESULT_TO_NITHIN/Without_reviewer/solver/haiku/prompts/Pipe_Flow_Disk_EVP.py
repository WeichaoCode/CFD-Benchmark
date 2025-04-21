import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

# Parameters
Nr = 100  # Number of radial points
Nphi = 64  # Number of azimuthal points
Re = 1e4   # Reynolds number
kz = 1.0   # Axial wavenumber
m = 1      # Azimuthal wavenumber

# Grid
r = np.linspace(0, 1, Nr)
dr = r[1] - r[0]
phi = np.linspace(0, 2*np.pi, Nphi)
dphi = phi[1] - phi[0]

# Background flow
w0 = 1 - r**2
dw0dr = -2*r

# Differential operators in r
def Dr_matrix(N):
    diagonals = [-1/2/dr, 0, 1/2/dr]
    positions = [-1, 0, 1]
    Dr = sparse.diags(diagonals, positions, shape=(N,N))
    Dr = Dr.tolil()
    Dr[0,0:3] = [-3/2/dr, 2/dr, -1/2/dr]  # Forward difference at r=0
    Dr[-1,-3:] = [1/2/dr, -2/dr, 3/2/dr]  # Backward difference at r=1
    return Dr.tocsr()

def Dr2_matrix(N):
    diagonals = [1/dr**2, -2/dr**2, 1/dr**2]
    positions = [-1, 0, 1]
    Dr2 = sparse.diags(diagonals, positions, shape=(N,N))
    Dr2 = Dr2.tolil()
    Dr2[0,0:3] = [2/dr**2, -5/dr**2, 4/dr**2]  # At r=0
    Dr2[-1,-3:] = [1/dr**2, -2/dr**2, 1/dr**2]  # At r=1
    return Dr2.tocsr()

# Build operators
Dr = Dr_matrix(Nr)
Dr2 = Dr2_matrix(Nr)

# Build system matrix
def build_matrix():
    # Initialize blocks
    N = Nr
    A11 = sparse.diags(1/r, 0, shape=(N,N)) @ Dr + Dr2 - sparse.diags(1/r**2, 0, shape=(N,N))
    A12 = sparse.diags(-m/r**2, 0, shape=(N,N))
    A21 = sparse.diags(m/r**2, 0, shape=(N,N))
    A22 = sparse.diags(1/r, 0, shape=(N,N)) @ Dr + Dr2 - sparse.diags(1/r**2, 0, shape=(N,N))
    A33 = sparse.diags(1/r, 0, shape=(N,N)) @ Dr + Dr2
    
    # Combine blocks
    Z = sparse.csr_matrix((N,N))
    row1 = sparse.hstack([A11, A12, Z])
    row2 = sparse.hstack([A21, A22, Z]) 
    row3 = sparse.hstack([Z, Z, A33])
    A = sparse.vstack([row1, row2, row3])
    
    # Add advection terms
    B = sparse.diags(w0, 0, shape=(N,N)) @ Dr
    C = sparse.diags(dw0dr, 0, shape=(N,N))
    
    # Full operator
    L = -1j*kz*B - (1/Re)*A + sparse.eye(3*N)
    
    return L.tocsr()

# Solve eigenvalue problem
L = build_matrix()
eigenvalues, eigenvectors = eigs(L, k=1, which='LR')

# Extract components
ur = eigenvectors[:Nr,0]
uphi = eigenvectors[Nr:2*Nr,0]
w = eigenvectors[2*Nr:,0]

# Normalize
norm = np.max(np.abs(w))
ur = ur/norm
uphi = uphi/norm 
w = w/norm

# Save results
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/ur_Pipe_Flow_Disk_EVP.npy', ur)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/uphi_Pipe_Flow_Disk_EVP.npy', uphi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/w_Pipe_Flow_Disk_EVP.npy', w)