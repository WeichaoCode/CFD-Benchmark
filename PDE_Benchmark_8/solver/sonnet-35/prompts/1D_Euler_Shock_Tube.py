import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 1.4
nx = 200  # Number of spatial points
dx = 2.0 / (nx - 1)  # Spatial step size
x = np.linspace(-1, 1, nx)

# Time parameters
t_start = 0.0
t_end = 0.25
dt = 0.0001  # Time step size
nt = int((t_end - t_start) / dt)

# Initial conditions
rho = np.zeros(nx)
rho[x < 0] = 1.0  # Left region
rho[x >= 0] = 0.125  # Right region

u = np.zeros(nx)  # Velocity is zero everywhere initially

p = np.zeros(nx)
p[x < 0] = 1.0  # Left region pressure
p[x >= 0] = 0.1  # Right region pressure

# Compute conservative variables
E = p / ((gamma - 1.0)) + 0.5 * u**2
U1 = rho
U2 = rho * u
U3 = rho * E

# Roe's approximate Riemann solver
def roe_flux(UL, UR):
    # Safety constants
    eps = 1e-12
    
    # Compute primitive variables with robust handling
    rhoL = max(UL[0], eps)
    rhoR = max(UR[0], eps)
    
    uL = np.clip(UL[1] / rhoL, -1e10, 1e10)
    uR = np.clip(UR[1] / rhoR, -1e10, 1e10)
    
    # Energy and pressure computation with robust handling
    EL = max(UL[2], eps)
    ER = max(UR[2], eps)
    
    pL = max((gamma-1)*(EL - 0.5*UL[1]**2/rhoL), eps)
    pR = max((gamma-1)*(ER - 0.5*UR[1]**2/rhoR), eps)
    
    # Roe averages with safety checks
    sqrt_rhoL = np.sqrt(max(rhoL, eps))
    sqrt_rhoR = np.sqrt(max(rhoR, eps))
    
    # Weighted average
    u_avg = (sqrt_rhoL*uL + sqrt_rhoR*uR) / (sqrt_rhoL + sqrt_rhoR)
    
    # Enthalpy computation
    HL = np.clip((EL + pL) / rhoL, -1e10, 1e10)
    HR = np.clip((ER + pR) / rhoR, -1e10, 1e10)
    
    H_avg = (sqrt_rhoL*HL + sqrt_rhoR*HR) / (sqrt_rhoL + sqrt_rhoR)
    
    # Sound speed with robust handling
    a_avg = np.sqrt(max((gamma-1)*(H_avg - 0.5*u_avg**2), eps))
    
    # Eigenvalues with entropy fix
    lambda1 = np.abs(u_avg - a_avg)
    lambda2 = np.abs(u_avg)
    lambda3 = np.abs(u_avg + a_avg)
    
    # Flux computation with clipping
    FL = np.array([
        np.clip(rhoL*uL, -1e10, 1e10),
        np.clip(rhoL*uL**2 + pL, -1e10, 1e10),
        np.clip(uL*(EL + pL), -1e10, 1e10)
    ])
    
    FR = np.array([
        np.clip(rhoR*uR, -1e10, 1e10),
        np.clip(rhoR*uR**2 + pR, -1e10, 1e10),
        np.clip(uR*(ER + pR), -1e10, 1e10)
    ])
    
    # Compute flux differences with clipping
    diff = np.clip(np.array([
        lambda1 * (UR[0] - UL[0]),
        lambda2 * (UR[1] - UL[1]),
        lambda3 * (UR[2] - UL[2])
    ]), -1e10, 1e10)
    
    return np.clip(0.5 * (FL + FR - diff), -1e10, 1e10)

# Time integration (Forward Euler)
for _ in range(nt):
    # Create conservative variable array
    U = np.column_stack((U1, U2, U3))
    
    # Compute fluxes
    F = np.zeros_like(U)
    for j in range(1, nx-1):
        # Compute flux at interface using Roe's method
        flux = roe_flux(U[j-1], U[j])
        F[j-1] = np.clip(F[j-1] + flux, -1e10, 1e10)
        F[j] = np.clip(F[j] - flux, -1e10, 1e10)
    
    # Update conservative variables
    U1 -= dt/dx * F[:, 0]
    U2 -= dt/dx * F[:, 1]
    U3 -= dt/dx * F[:, 2]
    
    # Apply reflective boundary conditions
    U1[0] = U1[1]
    U1[-1] = U1[-2]
    U2[0] = -U2[1]
    U2[-1] = -U2[-2]
    U3[0] = U3[1]
    U3[-1] = U3[-2]

# Compute final primitive variables
rho_final = np.clip(U1, 1e-12, 1e10)
u_final = np.clip(U2 / rho_final, -1e10, 1e10)
p_final = np.clip((gamma-1)*(U3 - 0.5*U2**2/rho_final), 1e-12, 1e10)

# Save variables
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/rho_final_1D_Euler_Shock_Tube.npy', rho_final)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/u_final_1D_Euler_Shock_Tube.npy', u_final)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/p_final_1D_Euler_Shock_Tube.npy', p_final)