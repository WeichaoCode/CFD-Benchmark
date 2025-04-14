import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem Parameters
Lx, Ly = 1.0, 1.0  # Domain dimensions
nx, ny = 30, 30  # Grid resolution
nu = 0.001  # Kinematic viscosity
dt = 0.005  # Time step
t_final = 0.5  # Final time

# Grid generation
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize fields
psi = np.zeros((ny, nx))
omega = np.zeros((ny, nx))

# Initial vortex layers
omega[int(ny*0.4):int(ny*0.6), int(nx*0.4):int(nx*0.6)] = 1.0

# Fast Poisson solver using sparse matrix
def solve_poisson(omega):
    # Create sparse Poisson matrix
    rows, cols, data = [], [], []
    
    for i in range(ny):
        for j in range(nx):
            row = i * nx + j
            
            # Main diagonal
            rows.append(row)
            cols.append(row)
            data.append(-4.0)
            
            # Neighboring diagonals
            if j > 0:  # Left
                rows.append(row)
                cols.append(row - 1)
                data.append(1.0)
            
            if j < nx - 1:  # Right
                rows.append(row)
                cols.append(row + 1)
                data.append(1.0)
            
            if i > 0:  # Up
                rows.append(row)
                cols.append(row - nx)
                data.append(1.0)
            
            if i < ny - 1:  # Down
                rows.append(row)
                cols.append(row + nx)
                data.append(1.0)
    
    # Create sparse matrix in CSR format
    A = sp.csr_matrix((data, (rows, cols)), shape=(nx*ny, nx*ny))
    
    # Right-hand side
    b = -omega.flatten() * dx * dy
    
    # Solve using sparse linear algebra
    psi_flat = spla.spsolve(A, b)
    return psi_flat.reshape((ny, nx))

# Compute velocities
def compute_velocities(psi):
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    
    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dy)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx)
    
    return u, v

# Periodic boundary conditions
def apply_periodic_bc(field):
    field[:, 0] = field[:, -2]
    field[:, -1] = field[:, 1]
    field[0, :] = field[-2, :]
    field[-1, :] = field[1, :]
    return field

# Time integration
u, v = np.zeros((ny, nx)), np.zeros((ny, nx))
time = 0
while time < t_final:
    # Solve Poisson equation
    psi = solve_poisson(omega)
    
    # Compute velocities
    u, v = compute_velocities(psi)
    
    # Vorticity transport
    omega_old = omega.copy()
    
    # Simplified transport equation
    omega[1:-1, 1:-1] = omega_old[1:-1, 1:-1] - dt * (
        u[1:-1, 1:-1] * (omega_old[1:-1, 2:] - omega_old[1:-1, :-2]) / (2*dx) +
        v[1:-1, 1:-1] * (omega_old[2:, 1:-1] - omega_old[:-2, 1:-1]) / (2*dy) +
        nu * ((omega_old[1:-1, 2:] - 2*omega_old[1:-1, 1:-1] + omega_old[1:-1, :-2]) / dx**2 +
              (omega_old[2:, 1:-1] - 2*omega_old[1:-1, 1:-1] + omega_old[:-2, 1:-1]) / dy**2)
    )
    
    # Apply periodic boundary conditions
    omega = apply_periodic_bc(omega)
    
    time += dt

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/psi_Vortex_Roll_Up.npy', psi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/omega_Vortex_Roll_Up.npy', omega)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_Vortex_Roll_Up.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/v_Vortex_Roll_Up.npy', v)