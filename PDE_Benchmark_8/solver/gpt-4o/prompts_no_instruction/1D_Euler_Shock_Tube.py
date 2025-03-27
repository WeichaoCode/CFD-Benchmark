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
U = np.array([rho, rho * u, rho * E])

# Time step calculation
a = np.sqrt(gamma * p / rho)
dt = CFL * dx / np.max(np.abs(u) + a)
t_final = 0.25
t = 0.0

# MacCormack method
while t < t_final:
    # Predictor step
    F = np.array([U[1], U[1]**2 / U[0] + (gamma - 1) * (U[2] - 0.5 * U[1]**2 / U[0]), 
                  U[1] * (U[2] + (gamma - 1) * (U[2] - 0.5 * U[1]**2 / U[0])) / U[0]])
    U_pred = U.copy()
    U_pred[:, :-1] = U[:, :-1] - dt / dx * (F[:, 1:] - F[:, :-1])

    # Corrector step
    F_pred = np.array([U_pred[1], U_pred[1]**2 / U_pred[0] + (gamma - 1) * (U_pred[2] - 0.5 * U_pred[1]**2 / U_pred[0]), 
                       U_pred[1] * (U_pred[2] + (gamma - 1) * (U_pred[2] - 0.5 * U_pred[1]**2 / U_pred[0])) / U_pred[0]])
    U[:, 1:] = 0.5 * (U[:, 1:] + U_pred[:, 1:] - dt / dx * (F_pred[:, 1:] - F_pred[:, :-1]))

    # Reflective boundary conditions
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]

    # Update time
    t += dt

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/U_1D_Euler_Shock_Tube.npy', U)