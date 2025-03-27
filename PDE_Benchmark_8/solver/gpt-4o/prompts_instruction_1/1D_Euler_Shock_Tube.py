import numpy as np

# Constants
gamma = 1.4
CFL = 1.0
L = 2.0
Nx = 81
x = np.linspace(-1, 1, Nx)
dx = x[1] - x[0]

# Initial conditions
rho_L, u_L, p_L = 1.0, 0.0, 1.0
rho_R, u_R, p_R = 0.125, 0.0, 0.1

# Initialize variables
rho = np.where(x < 0, rho_L, rho_R)
u = np.where(x < 0, u_L, u_R)
p = np.where(x < 0, p_L, p_R)
E = p / ((gamma - 1) * rho) + 0.5 * u**2

# Conservative variables
U = np.zeros((Nx, 3))
U[:, 0] = rho
U[:, 1] = rho * u
U[:, 2] = rho * E

# Time step calculation
a = np.sqrt(gamma * p / rho)
dt = CFL * dx / np.max(np.abs(u) + a)
t_final = 0.25
t = 0.0

# MacCormack method
while t < t_final:
    # Predictor step
    F = np.zeros((Nx, 3))
    F[:, 0] = rho * u
    F[:, 1] = rho * u**2 + p
    F[:, 2] = u * (rho * E + p)
    
    U_pred = np.copy(U)
    U_pred[:-1] = U[:-1] - dt / dx * (F[1:] - F[:-1])
    
    # Update primitive variables
    rho_pred = U_pred[:, 0]
    u_pred = U_pred[:, 1] / rho_pred
    E_pred = U_pred[:, 2] / rho_pred
    p_pred = (gamma - 1) * rho_pred * (E_pred - 0.5 * u_pred**2)
    
    # Corrector step
    F_pred = np.zeros((Nx, 3))
    F_pred[:, 0] = rho_pred * u_pred
    F_pred[:, 1] = rho_pred * u_pred**2 + p_pred
    F_pred[:, 2] = u_pred * (rho_pred * E_pred + p_pred)
    
    U[1:] = 0.5 * (U[1:] + U_pred[1:] - dt / dx * (F_pred[1:] - F_pred[:-1]))
    
    # Update primitive variables
    rho = U[:, 0]
    u = U[:, 1] / rho
    E = U[:, 2] / rho
    p = (gamma - 1) * rho * (E - 0.5 * u**2)
    
    # Reflective boundary conditions
    U[0] = U[1]
    U[-1] = U[-2]
    
    # Update time
    t += dt

# Save the final solution
final_solution = np.vstack((rho, u, p)).T
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_1/final_solution_1D_Euler_Shock_Tube.npy', final_solution)