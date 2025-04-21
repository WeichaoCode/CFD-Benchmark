import numpy as np
import scipy.linalg
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_stability(nr, kz, Re):
    """
    Solves the linearized incompressible Navier-Stokes equations in cylindrical coordinates
    for the stability of a pipe flow.

    Args:
        nr (int): Number of radial grid points.
        kz (float): Axial wavenumber.
        Re (float): Reynolds number.

    Returns:
        tuple: Eigenvalue with largest real part, u, w, and p as numpy arrays.
    """

    dr = 1.0 / (nr - 1)
    r = np.linspace(0, 1, nr)
    w0 = 1 - r**2
    dw0_dr = -2 * r

    # Create finite difference matrices
    def create_matrix(a, b, c, n):
        return diags([a, b, c], [-1, 0, 1], shape=(n, n), format="csc")

    # Radial derivatives
    Ar = create_matrix(-1 / (2 * dr), 0, 1 / (2 * dr), nr)
    Ar[0, 0] = 0
    Ar[0, 1] = 0
    Ar[-1, -2] = -1 / dr
    Ar[-1, -1] = 1 / dr

    A2r = create_matrix(1 / dr**2, -2 / dr**2, 1 / dr**2, nr)
    A2r[0, 0] = 0
    A2r[0, 1] = 0
    A2r[-1, -2] = 1 / dr**2
    A2r[-1, -1] = -1 / dr**2

    # Add curvature terms
    Ar_c = np.diag(1 / r) @ Ar
    A2r_c = np.diag(1 / r) @ Ar + np.diag(-1 / r**2)

    # Construct the linear operator (LHS of the eigenvalue problem)
    N = 3 * nr
    A = np.zeros((N, N), dtype=complex)

    # Continuity equation
    A[:nr, nr:2*nr] = Ar_c.toarray()
    A[:nr, 2*nr:] = 1j * kz * np.eye(nr)

    # Radial momentum equation
    A[nr:2*nr, nr:2*nr] = np.diag(w0 * 1j * kz) - (A2r_c + np.diag(-1 / r**2) + kz**2 * np.eye(nr)) / Re
    A[nr:2*nr, :nr] = np.diag(dw0_dr)
    A[nr:2*nr, 2*nr:] = Ar.toarray()

    # Axial momentum equation
    A[2*nr:, nr:2*nr] = np.diag(w0) @ Ar_c.toarray()
    A[2*nr:, 2*nr:] = np.diag(w0 * 1j * kz) - (A2r_c + kz**2 * np.eye(nr)) / Re
    A[2*nr:, :nr] = 1j * kz * np.diag(w0)
    A[2*nr:, 2*nr:] += 1j * kz * np.diag(w0)

    # Boundary conditions
    A[nr, :] = 0
    A[nr, nr] = 1
    A[2*nr, :] = 0
    A[2*nr, 2*nr] = 1
    A[N-1, :] = 0
    A[N-1, N-1] = 1

    # Solve the generalized eigenvalue problem
    B = np.eye(N)
    B[nr:2*nr, nr:2*nr] = -np.eye(nr)
    B[2*nr:, 2*nr:] = -np.eye(nr)

    eigenvalues, eigenvectors = scipy.linalg.eig(A, B)

    # Find the eigenvalue with the largest real part
    largest_eigenvalue = eigenvalues[np.argmax(eigenvalues.real)]

    # Extract the corresponding eigenvector
    eigenvector = eigenvectors[:, np.argmax(eigenvalues.real)]

    # Extract u, w, and p from the eigenvector
    u = eigenvector[:nr]
    w = eigenvector[nr:2*nr]
    p = eigenvector[2*nr:]

    return largest_eigenvalue, u, w, p

# Parameters
nr = 50  # Number of radial grid points
kz = 1.0  # Axial wavenumber
Re = 1e4  # Reynolds number

# Solve the stability problem
s, u, w, p = solve_stability(nr, kz, Re)

# Save the variables
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_Pipe_Flow_Disk_EVP.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/w_Pipe_Flow_Disk_EVP.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_Pipe_Flow_Disk_EVP.npy', p)