import numpy as np

# Parameters
gamma = 1.4
x_start, x_end = -1.0, 1.0
t_start, t_end = 0.0, 0.25
nx = 200  # Number of spatial points
dx = (x_end - x_start) / (nx - 1)
dt = 0.0005  # Time step
nt = int((t_end - t_start) / dt)  # Number of time steps

# Initial conditions
rho_L, u_L, p_L = 1.0, 0.0, 1.0
rho_R, u_R, p_R = 0.125, 0.0, 0.1

# Discretize the domain
x = np.linspace(x_start, x_end, nx)

# Initialize the conservative variables
U = np.zeros((3, nx))

# Set initial conditions
U[0, :nx//2] = rho_L
U[1, :nx//2] = rho_L * u_L
U[2, :nx//2] = p_L / (gamma - 1) + 0.5 * rho_L * u_L**2

U[0, nx//2:] = rho_R
U[1, nx//2:] = rho_R * u_R
U[2, nx//2:] = p_R / (gamma - 1) + 0.5 * rho_R * u_R**2

# Function to compute flux
def compute_flux(U):
    rho = U[0]
    rho_u = U[1]
    rho_E = U[2]
    
    u = rho_u / rho
    p = (gamma - 1) * (rho_E - 0.5 * rho * u**2)
    
    F = np.zeros_like(U)
    F[0] = rho_u
    F[1] = rho_u * u + p
    F[2] = u * (rho_E + p)
    
    return F

# Time-stepping loop
for n in range(nt):
    F = compute_flux(U)
    
    # Apply Lax-Friedrichs scheme
    U[:, 1:-1] = 0.5 * (U[:, :-2] + U[:, 2:]) - dt / (2 * dx) * (F[:, 2:] - F[:, :-2])
    
    # Reflective boundary conditions
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]

# Extract final values
rho_final = U[0]
u_final = U[1] / U[0]
p_final = (gamma - 1) * (U[2] - 0.5 * U[0] * u_final**2)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts/rho_final_1D_Euler_Shock_Tube.npy', rho_final)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts/u_final_1D_Euler_Shock_Tube.npy', u_final)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts/p_final_1D_Euler_Shock_Tube.npy', p_final)