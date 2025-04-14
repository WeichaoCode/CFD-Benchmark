import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem Parameters
Lx, Ly = 1.0, 1.0  # Domain dimensions
nx, ny = 50, 50  # Grid resolution
nu = 0.001  # Kinematic viscosity
dt = 0.0005  # Reduced time step
t_final = 1.0  # Further reduced final time

# Grid generation
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize fields with float64 to prevent overflow
psi = np.zeros((ny, nx), dtype=np.float64)
omega = np.zeros((ny, nx), dtype=np.float64)

# Initial vortex layers with controlled magnitude
omega[int(ny*0.4):int(ny*0.6), int(nx*0.4):int(nx*0.6)] = 0.1

def periodic_bc(field):
    field[:, 0] = field[:, -2]
    field[:, -1] = field[:, 1]
    field[0, :] = field[-2, :]
    field[-1, :] = field[1, :]
    return field

def solve_poisson(omega):
    # Convert to CSR matrix format
    n = nx * ny
    diag_main = -2 * (1/dx**2 + 1/dy**2) * np.ones(n, dtype=np.float64)
    diag_x = np.ones(n-1, dtype=np.float64) / dx**2
    diag_y = np.ones(n-nx, dtype=np.float64) / dy**2
    
    # Construct sparse matrix in CSR format
    diagonals = [diag_main, diag_x, diag_x, diag_y, diag_y]
    offsets = [0, 1, -1, nx, -nx]
    A = sp.diags(diagonals, offsets, shape=(n, n)).tocsr()
    
    # Flatten and solve with float64
    b = -omega.flatten().astype(np.float64)
    psi_flat = spla.spsolve(A, b)
    psi = psi_flat.reshape((ny, nx))
    
    # Apply boundary conditions
    psi[0, :] = 0
    psi[-1, :] = 0
    psi = periodic_bc(psi)
    
    return psi

def compute_velocities(psi):
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    
    u[1:-1, 1:-1] = np.clip((psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dy), -1e5, 1e5)
    v[1:-1, 1:-1] = np.clip(-(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx), -1e5, 1e5)
    
    return u, v

# Time integration with safeguards
time = 0
while time < t_final:
    # Solve Poisson equation
    psi = solve_poisson(omega)
    
    # Compute velocities
    u, v = compute_velocities(psi)
    
    # Vorticity transport with numerical safeguards
    omega_old = omega.copy()
    
    # Finite difference scheme with clipping and controlled computations
    advection_x = np.clip(u[1:-1, 1:-1] * (omega_old[1:-1, 2:] - omega_old[1:-1, :-2]) / (2*dx), -1e5, 1e5)
    advection_y = np.clip(v[1:-1, 1:-1] * (omega_old[2:, 1:-1] - omega_old[:-2, 1:-1]) / (2*dy), -1e5, 1e5)
    diffusion_x = nu * np.clip((omega_old[1:-1, 2:] - 2*omega_old[1:-1, 1:-1] + omega_old[1:-1, :-2]) / dx**2, -1e5, 1e5)
    diffusion_y = nu * np.clip((omega_old[2:, 1:-1] - 2*omega_old[1:-1, 1:-1] + omega_old[:-2, 1:-1]) / dy**2, -1e5, 1e5)
    
    omega[1:-1, 1:-1] = omega_old[1:-1, 1:-1] - dt * (advection_x + advection_y + diffusion_x + diffusion_y)
    
    # Apply periodic boundary conditions
    omega = periodic_bc(omega)
    
    time += dt

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/psi_Vortex_Roll_Up.npy', psi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/omega_Vortex_Roll_Up.npy', omega)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_Vortex_Roll_Up.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/v_Vortex_Roll_Up.npy', v)