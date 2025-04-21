#!/usr/bin/env python3
import numpy as np

# Parameters and constants
gamma = 1.4
x_start = -1.0
x_end = 1.0
t_end = 0.25
CFL = 0.5
nx = 400

dx = (x_end - x_start) / nx
# Create computational grid with ghost cells on both sides
# Domain indices 1...nx correspond to physical domain
x = np.linspace(x_start - dx, x_end + dx, nx + 2)

# Allocate conserved variable array U = [rho, rho*u, rho*E]
U = np.zeros((3, nx + 2))

# Set initial conditions for each cell in the physical domain (i=1,..,nx)
for i in range(1, nx+1):
    if x[i] < 0:
        # Left state
        rho = 1.0
        u = 0.0
        p = 1.0
    else:
        # Right state
        rho = 0.125
        u = 0.0
        p = 0.1
    # Total energy per unit mass: E = p/((gamma-1)*rho) + 0.5*u^2
    E = p / ((gamma - 1) * rho) + 0.5 * u**2
    U[0, i] = rho
    U[1, i] = rho * u
    U[2, i] = rho * E

def compute_flux(U):
    """
    Compute the flux vector F for the conserved variables U.
    U shape : (3, N) where N can be any number of cells.
    Returns F with the same shape.
    """
    rho = U[0]
    u = U[1] / rho
    # Compute pressure: p = (gamma-1) * (rho*E - 0.5*rho*u^2)
    p = (gamma - 1) * (U[2] - 0.5 * rho * u**2)
    F = np.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = u * (U[2] + p)
    return F

def apply_reflective_BC(U):
    """
    Apply reflective (no-flux) boundary conditions.
    Left ghost cell from first physical cell: rho mirrored, momentum reversed, energy same.
    Right ghost cell from last physical cell.
    """
    # Left BC (i=0) from i=1
    U[0, 0] = U[0, 1]
    U[1, 0] = -U[1, 1]
    U[2, 0] = U[2, 1]
    # Right BC (i=-1) from i=-2
    U[0, -1] = U[0, -2]
    U[1, -1] = -U[1, -2]
    U[2, -1] = U[2, -2]
    return U

# Initialize ghost cells
U = apply_reflective_BC(U)

t = 0.0
while t < t_end:
    # Compute primitive variables for CFL condition (only physical domain cells)
    rho = U[0, 1:-1]
    u = U[1, 1:-1] / rho
    p = (gamma - 1) * (U[2, 1:-1] - 0.5 * rho * u**2)
    a = np.sqrt(np.abs(gamma * p / rho))
    max_speed = np.max(np.abs(u) + a)
    dt = CFL * dx / max_speed
    if t + dt > t_end:
        dt = t_end - t

    # Compute fluxes at all cells
    F = compute_flux(U)

    # Lax-Friedrichs scheme update for physical cells: i = 1 ... nx
    U_new = U.copy()
    # Use vectorized update for interior cells
    U_new[:, 1:-1] = 0.5 * (U[:, 2:] + U[:, :-2]) - (dt / (2 * dx)) * (F[:, 2:] - F[:, :-2])

    # Update the solution array
    U = U_new.copy()

    # Apply reflective boundary conditions
    U = apply_reflective_BC(U)

    t += dt

# Final solution in the physical domain, indices 1 to nx (1D arrays)
rho_final = U[0, 1:-1]
rho_u_final = U[1, 1:-1]
rho_E_final = U[2, 1:-1]

# Save final solutions as .npy files
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/rho_final_1D_Euler_Shock_Tube.npy', rho_final)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/rho_u_final_1D_Euler_Shock_Tube.npy', rho_u_final)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/rho_E_final_1D_Euler_Shock_Tube.npy', rho_E_final)