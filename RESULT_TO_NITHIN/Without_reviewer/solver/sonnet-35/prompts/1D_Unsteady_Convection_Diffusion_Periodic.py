import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
L = 2.0  # Domain length
u = 0.2  # Fluid velocity
rho = 1.0  # Density
gamma = 0.001  # Diffusion coefficient
t_end = 2.5  # Total simulation time
m = 0.5  # Gaussian peak center
s = 0.1  # Gaussian peak width

# Discretization parameters
nx = 100  # Number of spatial points
nt = 500  # Number of time steps

# Grid generation
dx = L / (nx - 1)
x = np.linspace(0, L, nx)
dt = t_end / (nt - 1)

# Initialize solution array
phi = np.zeros(nx)

# Initial condition (Gaussian profile)
phi = np.exp(-((x - m)/s)**2)

# Finite Volume Method
for n in range(nt):
    # Create temporary array for new solution
    phi_new = np.zeros_like(phi)
    
    # Loop through interior points
    for i in range(nx):
        # Periodic boundary handling
        ip = (i + 1) % nx
        im = (i - 1 + nx) % nx
        
        # Convective flux (upwind)
        F_conv = u * phi[i]
        
        # Diffusive flux 
        F_diff = gamma * (phi[ip] - phi[i]) / dx
        
        # Update solution using finite volume method
        phi_new[i] = phi[i] - (dt/dx) * (F_conv - F_diff)
    
    # Update solution
    phi = phi_new.copy()

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/phi_1D_Unsteady_Convection_Diffusion_Periodic.npy', phi)