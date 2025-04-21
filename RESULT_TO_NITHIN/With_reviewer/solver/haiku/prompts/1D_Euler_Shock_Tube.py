import numpy as np

# Parameters
gamma = 1.4
nx = 400  # Number of spatial points
dx = 2.0/nx  # Spatial step size
x = np.linspace(-1, 1, nx)
cfl = 0.8
t_final = 0.25

# Initialize conservative variables
rho = np.ones(nx)
u = np.zeros(nx)
p = np.ones(nx)
E = p/(gamma-1)/rho + 0.5*u**2

# Set initial conditions
rho[x >= 0] = 0.125
p[x >= 0] = 0.1
E[x >= 0] = p[x >= 0]/(gamma-1)/rho[x >= 0] + 0.5*u[x >= 0]**2

# Initialize conservative vector U
U = np.zeros((3, nx))
U[0] = rho
U[1] = rho*u
U[2] = rho*E

def compute_flux(U):
    # Compute primitive variables
    rho = np.maximum(U[0], 1e-6)
    u = np.clip(U[1]/rho, -1e3, 1e3)  # Limit velocity
    E = np.clip(U[2]/rho, 0, 1e3)     # Limit energy
    p = np.maximum((gamma-1)*rho*(E - 0.5*u**2), 1e-6)
    
    # Compute flux vector
    F = np.zeros_like(U)
    F[0] = rho*u
    F[1] = rho*u**2 + p
    F[2] = np.clip(u*(rho*E + p), -1e6, 1e6)  # Limit energy flux
    return F

def compute_timestep(U):
    rho = np.maximum(U[0], 1e-6)
    u = np.clip(U[1]/rho, -1e3, 1e3)
    E = np.clip(U[2]/rho, 0, 1e3)
    p = np.maximum((gamma-1)*rho*(E - 0.5*u**2), 1e-6)
    c = np.sqrt(gamma*p/rho)
    return cfl*dx/np.max(np.abs(u) + c)

# Time integration
t = 0
while t < t_final:
    dt = compute_timestep(U)
    if t + dt > t_final:
        dt = t_final - t
        
    # Compute fluxes
    F = compute_flux(U)
    
    # Update solution using first-order upwind scheme
    F_plus = np.zeros_like(F)
    F_minus = np.zeros_like(F)
    
    # Split fluxes
    for i in range(3):
        F_plus[i,:-1] = np.maximum(F[i,:-1], 0)
        F_plus[i,-1] = F_plus[i,-2]
        F_minus[i,1:] = np.minimum(F[i,1:], 0)
        F_minus[i,0] = F_minus[i,1]
    
    # Update conservative variables
    dF = np.clip(F_plus - np.roll(F_plus, 1, axis=1) + F_minus - np.roll(F_minus, 1, axis=1), -1e6, 1e6)
    U = U - dt/dx*dF
    
    # Ensure physical values
    U[0] = np.maximum(U[0], 1e-6)
    U[1] = np.clip(U[1], -1e6, 1e6)
    E_min = np.maximum(1e-6, 0.5*U[1]**2/U[0]**2)
    U[2] = np.maximum(U[2], E_min*U[0])
    
    t += dt

# Compute final primitive variables
rho = U[0]
u = U[1]/U[0]
p = (gamma-1)*(U[2] - 0.5*rho*u**2)

# Save results
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/rho_1D_Euler_Shock_Tube.npy', rho)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/u_1D_Euler_Shock_Tube.npy', u)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/p_1D_Euler_Shock_Tube.npy', p)