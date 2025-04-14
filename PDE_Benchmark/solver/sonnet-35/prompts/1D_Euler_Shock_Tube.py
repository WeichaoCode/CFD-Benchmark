import numpy as np

# Problem parameters
gamma = 1.4  # Ratio of specific heats
x_min, x_max = -1.0, 1.0  # Domain
t_min, t_max = 0.0, 0.25  # Time domain
nx = 200  # Number of spatial points
nt = 200  # Number of time steps

# Grid setup
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)
dt = (t_max - t_min) / nt

# Initial conditions
rho = np.zeros(nx)
rho[x < 0] = 1.0  # Left region
rho[x >= 0] = 0.125  # Right region

u = np.zeros(nx)  # Velocity 

p = np.zeros(nx)
p[x < 0] = 1.0  # Left region 
p[x >= 0] = 0.1  # Right region

# Compute conservative variables
E = p / ((gamma - 1.0)) + 0.5 * u**2
U = np.zeros((3, nx))
U[0, :] = rho
U[1, :] = rho * u
U[2, :] = rho * E

# Roe's Riemann solver
def roe_flux(UL, UR, pL, pR):
    # Compute Roe average states
    rho_L = max(UL[0], 1e-15)
    rho_R = max(UR[0], 1e-15)
    
    u_L = np.clip(UL[1] / rho_L, -1e10, 1e10)
    u_R = np.clip(UR[1] / rho_R, -1e10, 1e10)
    
    # Compute enthalpies
    H_L = np.clip((UL[2] + pL) / rho_L, -1e10, 1e10)
    H_R = np.clip((UR[2] + pR) / rho_R, -1e10, 1e10)
    
    # Roe average
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    
    u_avg = np.clip((sqrt_rho_L * u_L + sqrt_rho_R * u_R) / 
                    (sqrt_rho_L + sqrt_rho_R), -1e10, 1e10)
    H_avg = np.clip((sqrt_rho_L * H_L + sqrt_rho_R * H_R) / 
                    (sqrt_rho_L + sqrt_rho_R), -1e10, 1e10)
    
    # Eigenvalues
    a_avg = np.sqrt(max(np.abs((gamma - 1.0) * (H_avg - 0.5 * u_avg**2)), 1e-15))
    lambda1 = u_avg - a_avg
    lambda2 = u_avg
    lambda3 = u_avg + a_avg
    
    # Compute flux
    F_L = np.array([
        np.clip(rho_L * u_L, -1e10, 1e10), 
        np.clip(rho_L * u_L**2 + pL, -1e10, 1e10), 
        np.clip(u_L * (UL[2] + pL), -1e10, 1e10)
    ])
    F_R = np.array([
        np.clip(rho_R * u_R, -1e10, 1e10), 
        np.clip(rho_R * u_R**2 + pR, -1e10, 1e10), 
        np.clip(u_R * (UR[2] + pR), -1e10, 1e10)
    ])
    
    # Compute flux differences with clipping
    diff = np.clip(UR - UL, -1e10, 1e10)
    
    return np.clip(0.5 * (F_L + F_R - 
                   np.abs(lambda1) * diff - 
                   np.abs(lambda2) * diff - 
                   np.abs(lambda3) * diff), -1e10, 1e10)

# Time integration (Forward Euler)
for _ in range(nt):
    # Compute fluxes
    F = np.zeros((3, nx))
    for j in range(1, nx):
        UL = U[:, j-1]
        UR = U[:, j]
        pL = p[j-1]
        pR = p[j]
        F[:, j] = roe_flux(UL, UR, pL, pR)
    
    # Update conservative variables with clipping
    U[:, 1:] = np.clip(U[:, 1:] - dt/dx * (F[:, 1:] - F[:, :-1]), 
                       -1e10, 1e10)
    
    # Update pressure with clipping
    p = np.clip((gamma - 1.0) * (U[2, :] - 0.5 * U[1, :]**2 / 
                np.clip(U[0, :], 1e-15, 1e10)), 0, 1e10)
    
    # Reflective boundary conditions
    U[1, 0] = -U[1, 1]
    U[1, -1] = -U[1, -2]

# Extract final solution
rho_final = np.clip(U[0, :], 0, 1e10)
u_final = np.clip(U[1, :] / np.clip(U[0, :], 1e-15, 1e10), -1e10, 1e10)
p_final = np.clip((gamma - 1.0) * (U[2, :] - 0.5 * U[1, :]**2 / 
                   np.clip(U[0, :], 1e-15, 1e10)), 0, 1e10)

# Save results
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/rho_final_1D_Euler_Shock_Tube.npy', rho_final)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_final_1D_Euler_Shock_Tube.npy', u_final)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/p_final_1D_Euler_Shock_Tube.npy', p_final)