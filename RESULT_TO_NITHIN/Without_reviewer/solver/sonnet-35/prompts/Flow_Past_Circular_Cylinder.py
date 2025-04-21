import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
r_min, r_max = 0.5, 10.0
theta_min, theta_max = 0, 2*np.pi
nu = 0.005
v_inf = 1.0

# Numerical grid parameters
nr, ntheta = 100, 100
nt = 1000  # time steps
dt = 0.01  # time step size

# Create grid
r = np.linspace(r_min, r_max, nr)
theta = np.linspace(theta_min, theta_max, ntheta)
R, Theta = np.meshgrid(r, theta)

# Initialize arrays
psi = np.zeros((ntheta, nr))
omega = np.zeros((ntheta, nr))

# Boundary conditions
def apply_boundary_conditions(psi, omega):
    # Inner boundary (cylinder surface)
    psi[0, :] = 20
    omega[0, :] = 2 * (20 - psi[0, :]) / (r[1] - r[0])**2
    
    # Outer boundary 
    psi[-1, :] = v_inf * R[-1, :] + 20
    omega[-1, :] = 0
    
    # Periodic boundary in theta
    psi[0, :] = psi[-1, :]
    omega[0, :] = omega[-1, :]
    
    return psi, omega

# Time-stepping using finite difference method
for n in range(nt):
    # Create copy of current solution
    psi_old = psi.copy()
    omega_old = omega.copy()
    
    # Compute velocity components
    u_r = np.zeros_like(psi)
    u_theta = np.zeros_like(psi)
    
    for i in range(1, ntheta-1):
        for j in range(1, nr-1):
            # Velocity in r direction
            u_r[i,j] = (1/R[i,j]) * (psi[i+1,j] - psi[i-1,j]) / (theta[1] - theta[0])
            
            # Velocity in theta direction 
            u_theta[i,j] = -(psi[i,j+1] - psi[i,j-1]) / (r[1] - r[0])
    
    # Compute vorticity transport
    for i in range(1, ntheta-1):
        for j in range(1, nr-1):
            # Finite difference discretization of vorticity transport
            d2omega_dr2 = (omega_old[i,j+1] - 2*omega_old[i,j] + omega_old[i,j-1]) / (r[1] - r[0])**2
            d2omega_dtheta2 = (omega_old[i+1,j] - 2*omega_old[i,j] + omega_old[i-1,j]) / (theta[1] - theta[0])**2
            
            # Vorticity transport equation 
            omega[i,j] = omega_old[i,j] + dt * (
                nu * (d2omega_dr2 + d2omega_dtheta2) - 
                u_r[i,j] * (omega_old[i,j+1] - omega_old[i,j-1]) / (r[1] - r[0]) -
                u_theta[i,j] * (omega_old[i+1,j] - omega_old[i-1,j]) / (theta[1] - theta[0])
            )
    
    # Apply boundary conditions
    psi, omega = apply_boundary_conditions(psi, omega)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/psi_Flow_Past_Circular_Cylinder.npy', psi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/omega_Flow_Past_Circular_Cylinder.npy', omega)