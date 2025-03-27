import numpy as np

# Constants
gamma = 1.4
CFL = 1.0
L = 2.0
Nx = 81
dx = L / (Nx - 1)
x = np.linspace(-1, 1, Nx)

# Initial conditions
rho = np.where(x < 0, 1.0, 0.125)
u = np.zeros(Nx)
p = np.where(x < 0, 1.0, 0.1)

# Compute initial conservative variables
E = p / ((gamma - 1) * rho) + 0.5 * u**2
U = np.array([rho, rho * u, rho * E])

# Time stepping parameters
t_final = 0.25
t = 0.0

# Function to compute flux vector
def compute_flux(U):
    rho = U[0]
    rho_u = U[1]
    rho_E = U[2]
    u = rho_u / rho
    p = (gamma - 1) * (rho_E - 0.5 * rho * u**2)
    F = np.array([rho_u, rho_u * u + p, u * (rho_E + p)])
    return F

# Time-stepping loop
while t < t_final:
    # Compute time step
    u = U[1] / U[0]
    p = (gamma - 1) * (U[2] - 0.5 * U[0] * u**2)
    c = np.sqrt(gamma * p / U[0])
    dt = CFL * dx / np.max(np.abs(u) + c)
    if t + dt > t_final:
        dt = t_final - t

    # Predictor step
    F = compute_flux(U)
    U_pred = U.copy()
    U_pred[:, :-1] = U[:, :-1] - dt / dx * (F[:, 1:] - F[:, :-1])

    # Apply reflective boundary conditions
    U_pred[:, 0] = U_pred[:, 1]
    U_pred[:, -1] = U_pred[:, -2]

    # Corrector step
    F_pred = compute_flux(U_pred)
    U[:, 1:-1] = 0.5 * (U[:, 1:-1] + U_pred[:, 1:-1] - dt / dx * (F_pred[:, 1:-1] - F_pred[:, :-2]))

    # Apply reflective boundary conditions
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]

    # Update time
    t += dt

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/U_1D_Euler_Shock_Tube.npy', U)