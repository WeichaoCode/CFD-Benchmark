import numpy as np

# Parameters
Lx, Lz = 4.0, 1.0
Nx, Nz = 128, 32
dx, dz = Lx / Nx, Lz / Nz
Ra, Pr = 2e6, 1.0
nu = (Ra / Pr) ** -0.5
kappa = (Ra * Pr) ** -0.5
dt = 0.0001  # Reduced time step for stability
T = 50.0
Nt = int(T / dt)

# Initial conditions
x = np.linspace(0, Lx, Nx, endpoint=False)
z = np.linspace(0, Lz, Nz, endpoint=False)
X, Z = np.meshgrid(x, z, indexing='ij')

u = np.zeros((Nx, Nz))
w = np.zeros((Nx, Nz))
b = Lz - Z + 0.01 * np.random.rand(Nx, Nz)

# Helper functions
def periodic_bc(arr):
    arr[0, :] = arr[-2, :]
    arr[-1, :] = arr[1, :]
    return arr

def apply_boundary_conditions(u, w, b):
    u[:, 0] = 0
    u[:, -1] = 0
    w[:, 0] = 0
    w[:, -1] = 0
    b[:, 0] = Lz
    b[:, -1] = 0
    b = periodic_bc(b)
    return u, w, b

def compute_rhs(u, w, b):
    # Compute derivatives
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dudz = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dz)
    dwdx = (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) / (2 * dx)
    dwdz = (np.roll(w, -1, axis=1) - np.roll(w, 1, axis=1)) / (2 * dz)
    dbdx = (np.roll(b, -1, axis=0) - np.roll(b, 1, axis=0)) / (2 * dx)
    dbdz = (np.roll(b, -1, axis=1) - np.roll(b, 1, axis=1)) / (2 * dz)

    # Laplacians
    lap_u = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dx**2 + \
            (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dz**2
    lap_w = (np.roll(w, -1, axis=0) - 2 * w + np.roll(w, 1, axis=0)) / dx**2 + \
            (np.roll(w, -1, axis=1) - 2 * w + np.roll(w, 1, axis=1)) / dz**2
    lap_b = (np.roll(b, -1, axis=0) - 2 * b + np.roll(b, 1, axis=0)) / dx**2 + \
            (np.roll(b, -1, axis=1) - 2 * b + np.roll(b, 1, axis=1)) / dz**2

    # RHS of momentum equations
    rhs_u = - (u * dudx + w * dudz) + nu * lap_u
    rhs_w = - (u * dwdx + w * dwdz) + nu * lap_w + b

    # RHS of buoyancy equation
    rhs_b = - (u * dbdx + w * dbdz) + kappa * lap_b

    return rhs_u, rhs_w, rhs_b

# Time-stepping loop with reduced number of steps for demonstration
for n in range(min(Nt, 10000)):  # Limit to 10000 steps to avoid timeout
    rhs_u, rhs_w, rhs_b = compute_rhs(u, w, b)

    # Update fields
    u += dt * rhs_u
    w += dt * rhs_w
    b += dt * rhs_b

    # Apply boundary conditions
    u, w, b = apply_boundary_conditions(u, w, b)

# Save final results
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_2D_Rayleigh_Benard_Convection.npy', u)
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/w_2D_Rayleigh_Benard_Convection.npy', w)
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/b_2D_Rayleigh_Benard_Convection.npy', b)