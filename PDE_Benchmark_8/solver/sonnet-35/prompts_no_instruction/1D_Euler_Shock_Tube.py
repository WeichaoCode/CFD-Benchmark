import numpy as np

# Problem parameters
gamma = 1.4
Lx = 2.0
N_x = 81
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 0.25
CFL = 1.0

# Initial conditions
rho_L, u_L, p_L = 1.0, 0.0, 1.0
rho_R, u_R, p_R = 0.125, 0.0, 0.1

# Grid setup
x = np.linspace(x_min, x_max, N_x)
dx = x[1] - x[0]

# Initialize conservative variables
U = np.zeros((N_x, 3))

# Set initial conditions
E_L = p_L / ((gamma - 1.0)) + 0.5 * u_L**2
E_R = p_R / ((gamma - 1.0)) + 0.5 * u_R**2

for i in range(N_x):
    if x[i] < 0:
        U[i, 0] = rho_L
        U[i, 1] = rho_L * u_L
        U[i, 2] = rho_L * E_L
    else:
        U[i, 0] = rho_R
        U[i, 1] = rho_R * u_R
        U[i, 2] = rho_R * E_R

# Compute pressure 
def compute_pressure(U):
    rho = U[:, 0]
    rho_u = U[:, 1]
    rho_E = U[:, 2]
    u = rho_u / rho
    E = rho_E / rho
    p = (gamma - 1.0) * rho * (E - 0.5 * u**2)
    return p

# Compute flux
def compute_flux(U):
    rho = U[:, 0]
    rho_u = U[:, 1]
    rho_E = U[:, 2]
    
    u = rho_u / rho
    p = compute_pressure(U)
    E = rho_E / rho
    
    F = np.zeros_like(U)
    F[:, 0] = rho_u
    F[:, 1] = rho_u**2 / rho + p
    F[:, 2] = u * (rho_E + p)
    
    return F

# MacCormack method
t = t_min
while t < t_max:
    # Compute max wave speed for time step
    p = compute_pressure(U)
    rho = U[:, 0]
    u = U[:, 1] / rho
    c = np.sqrt(gamma * p / rho)
    max_wave_speed = np.max(np.abs(u) + c)
    
    # Compute time step using CFL
    dt = CFL * dx / max_wave_speed
    if t + dt > t_max:
        dt = t_max - t
    
    # Predictor step
    F = compute_flux(U)
    U_pred = np.copy(U)
    U_pred[1:-1] = U[1:-1] - dt/dx * (F[2:] - F[1:-1])
    
    # Corrector step
    F_pred = compute_flux(U_pred)
    U[1:-1] = 0.5 * (U[1:-1] + U_pred[1:-1] - dt/dx * (F_pred[1:-1] - F_pred[:-2]))
    
    # Reflective boundary conditions
    U[0] = U[1]
    U[-1] = U[-2]
    
    t += dt

# Compute primitive variables for saving
rho = U[:, 0]
u = U[:, 1] / rho
p = compute_pressure(U)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/rho_1D_Euler_Shock_Tube.npy', rho)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_1D_Euler_Shock_Tube.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/p_1D_Euler_Shock_Tube.npy', p)