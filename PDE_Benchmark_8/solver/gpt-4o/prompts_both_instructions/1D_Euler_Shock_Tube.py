import numpy as np

# Constants
gamma = 1.4
CFL = 1.0
L = 2.0
Nx = 81
x = np.linspace(-1, 1, Nx)
dx = x[1] - x[0]

# Initial conditions
rho = np.where(x < 0, 1.0, 0.125)
u = np.zeros(Nx)
p = np.where(x < 0, 1.0, 0.1)

# Compute initial conservative variables
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

    # Apply reflective boundary conditions
    U_pred[:, 0] = U_pred[:, 1]
    U_pred[:, -1] = U_pred[:, -2]

    # Corrector step
    F_pred = np.array([U_pred[1], U_pred[1]**2 / U_pred[0] + (gamma - 1) * (U_pred[2] - 0.5 * U_pred[1]**2 / U_pred[0]), 
                       U_pred[1] * (U_pred[2] + (gamma - 1) * (U_pred[2] - 0.5 * U_pred[1]**2 / U_pred[0])) / U_pred[0]])
    U[:, 1:] = 0.5 * (U[:, 1:] + U_pred[:, 1:] - dt / dx * (F_pred[:, 1:] - F_pred[:, :-1]))

    # Apply reflective boundary conditions
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]

    # Update time
    t += dt

# Convert back to primitive variables
rho = U[0]
u = U[1] / rho
E = U[2] / rho
p = (gamma - 1) * (E - 0.5 * u**2) * rho

# Save the final solution
final_solution = np.array([rho, u, p])
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/final_solution_1D_Euler_Shock_Tube.npy', final_solution)