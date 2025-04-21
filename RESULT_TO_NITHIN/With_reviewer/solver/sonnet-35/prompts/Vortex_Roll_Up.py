import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
Lx, Ly = 1.0, 1.0  # Domain dimensions
nx, ny = 100, 100  # Grid points
nu = 0.001  # Kinematic viscosity
dt = 0.001  # Time step
t_final = 10.0  # Final simulation time

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

# Periodic boundary condition helper functions
def periodic_bc_x(field):
    field[:, 0] = field[:, -2]
    field[:, -1] = field[:, 1]
    return field

def periodic_bc_y(field):
    field[0, :] = field[-2, :]
    field[-1, :] = field[1, :]
    return field

# Finite difference method for solving vorticity-streamfunction equations
def solve_vorticity_streamfunction(omega, psi):
    # Poisson solver for streamfunction
    for _ in range(100):  # Jacobi iterations
        psi_old = psi.copy()
        
        # Finite difference Poisson equation
        psi[1:-1, 1:-1] = 0.25 * (
            psi_old[1:-1, 2:] + 
            psi_old[1:-1, :-2] + 
            psi_old[2:, 1:-1] + 
            psi_old[:-2, 1:-1] - 
            dx*dy*omega[1:-1, 1:-1]
        )
        
        # Enforce boundary conditions
        psi = periodic_bc_x(psi)
        psi[0, :] = 0  # Bottom boundary
        psi[-1, :] = 0  # Top boundary
    
    return psi

# Time integration
time = 0
while time < t_final:
    # Store old values
    omega_old = omega.copy()
    
    # Compute velocities from streamfunction
    u = np.zeros_like(omega)
    v = np.zeros_like(omega)
    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dy)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx)
    
    # Vorticity transport equation (finite difference)
    # Advection terms
    adv_x = u * np.gradient(omega_old, dx, axis=1)
    adv_y = v * np.gradient(omega_old, dy, axis=0)
    
    # Diffusion term
    diff_x = nu * np.gradient(np.gradient(omega_old, dx, axis=1), dx, axis=1)
    diff_y = nu * np.gradient(np.gradient(omega_old, dy, axis=0), dy, axis=0)
    
    # Update vorticity
    omega[1:-1, 1:-1] = omega_old[1:-1, 1:-1] - dt * (adv_x[1:-1, 1:-1] + adv_y[1:-1, 1:-1]) + dt * (diff_x[1:-1, 1:-1] + diff_y[1:-1, 1:-1])
    
    # Enforce periodic boundary conditions
    omega = periodic_bc_x(omega)
    omega = periodic_bc_y(omega)
    
    # Solve for streamfunction
    psi = solve_vorticity_streamfunction(omega, psi)
    
    # Update time
    time += dt

# Save final solutions
np.save('/PDE_Benchmark/results/prediction/sonnet-35/prompts/psi_Vortex_Roll_Up.npy', psi)
np.save('/PDE_Benchmark/results/prediction/sonnet-35/prompts/omega_Vortex_Roll_Up.npy', omega)