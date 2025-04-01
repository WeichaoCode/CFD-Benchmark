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

# Conservative variables
E = p / ((gamma - 1) * rho) + 0.5 * u**2
U = np.zeros((Nx, 3))
U[:, 0] = rho
U[:, 1] = rho * u
U[:, 2] = rho * E

# Time stepping
t = 0.0
t_end = 0.25

while t < t_end:
    # Calculate fluxes
    F = np.zeros((Nx, 3))
    F[:, 0] = U[:, 1]
    F[:, 1] = U[:, 1]**2 / U[:, 0] + (gamma - 1) * (U[:, 2] - 0.5 * U[:, 1]**2 / U[:, 0])
    F[:, 2] = (U[:, 2] + (gamma - 1) * (U[:, 2] - 0.5 * U[:, 1]**2 / U[:, 0])) * U[:, 1] / U[:, 0]

    # Calculate time step
    a = np.sqrt(gamma * (gamma - 1) * (U[:, 2] - 0.5 * U[:, 1]**2 / U[:, 0]) / U[:, 0])
    dt = CFL * dx / np.max(np.abs(U[:, 1] / U[:, 0]) + a)
    if t + dt > t_end:
        dt = t_end - t

    # Predictor step
    U_pred = np.copy(U)
    U_pred[:-1] -= dt / dx * (F[1:] - F[:-1])

    # Reflective boundary conditions
    U_pred[0] = U_pred[1]
    U_pred[-1] = U_pred[-2]

    # Recalculate fluxes for predicted step
    F_pred = np.zeros((Nx, 3))
    F_pred[:, 0] = U_pred[:, 1]
    F_pred[:, 1] = U_pred[:, 1]**2 / U_pred[:, 0] + (gamma - 1) * (U_pred[:, 2] - 0.5 * U_pred[:, 1]**2 / U_pred[:, 0])
    F_pred[:, 2] = (U_pred[:, 2] + (gamma - 1) * (U_pred[:, 2] - 0.5 * U_pred[:, 1]**2 / U_pred[:, 0])) * U_pred[:, 1] / U_pred[:, 0]

    # Corrector step
    U[1:] = 0.5 * (U[1:] + U_pred[1:] - dt / dx * (F_pred[1:] - F_pred[:-1]))

    # Reflective boundary conditions
    U[0] = U[1]
    U[-1] = U[-2]

    # Update time
    t += dt

# Extract final values
rho = U[:, 0]
u = U[:, 1] / U[:, 0]
E = U[:, 2] / U[:, 0]
p = (gamma - 1) * (U[:, 2] - 0.5 * U[:, 1]**2 / U[:, 0])

# Save final values
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/rho_1D_Euler_Shock_Tube.npy', rho)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_1D_Euler_Shock_Tube.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/p_1D_Euler_Shock_Tube.npy', p)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/E_1D_Euler_Shock_Tube.npy', E)