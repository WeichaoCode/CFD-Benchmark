import numpy as np

# Parameters
Lx, Ly = 1.0, 1.0  # Domain size
Nx, Ny = 128, 128  # Number of grid points
dx, dy = Lx / Nx, Ly / Ny
nu = 0.001  # Kinematic viscosity
dt = 0.001  # Time step
T = 1.0  # Final time
nt = int(T / dt)  # Number of time steps

# Initialize fields
psi = np.zeros((Nx, Ny))
omega = np.zeros((Nx, Ny))

# Initial condition: pair of vortex layers
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')
omega = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

# Helper functions
def laplacian(f):
    return (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx**2 + \
           (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dy**2

def periodic_bc(f):
    f[0, :] = f[-2, :]
    f[-1, :] = f[1, :]
    return f

# Time-stepping loop
for n in range(nt):
    # Solve Poisson equation for streamfunction
    for _ in range(50):  # Simple iterative solver
        psi[1:-1, 1:-1] = 0.25 * (psi[2:, 1:-1] + psi[:-2, 1:-1] +
                                  psi[1:-1, 2:] + psi[1:-1, :-2] +
                                  dx**2 * omega[1:-1, 1:-1])
        psi[:, 0] = 0  # Dirichlet BC
        psi[:, -1] = 0  # Dirichlet BC
        psi = periodic_bc(psi)

    # Compute velocity field
    u = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * dy)
    v = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * dx)

    # Update vorticity using the vorticity transport equation
    omega[1:-1, 1:-1] += dt * (
        - u[1:-1, 1:-1] * (np.roll(omega, -1, axis=0)[1:-1, 1:-1] - np.roll(omega, 1, axis=0)[1:-1, 1:-1]) / (2 * dx)
        - v[1:-1, 1:-1] * (np.roll(omega, -1, axis=1)[1:-1, 1:-1] - np.roll(omega, 1, axis=1)[1:-1, 1:-1]) / (2 * dy)
        + nu * laplacian(omega)[1:-1, 1:-1]
    )
    omega = periodic_bc(omega)

# Save final results
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/psi_Vortex_Roll_Up.npy', psi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/omega_Vortex_Roll_Up.npy', omega)