import numpy as np

# Parameters
L = 2.0  # Domain length
nx = 200  # Number of cells
dx = L/nx
x = np.linspace(dx/2, L-dx/2, nx)  # Cell centers
dt = 0.001  # Time step
t_final = 2.5
nt = int(t_final/dt)

# Physical parameters
u = 0.2  # Velocity
rho = 1.0  # Density 
gamma = 0.001  # Diffusion coefficient
D = gamma/rho  # Diffusivity

# Initial condition
m = 0.5  # Mean of Gaussian
s = 0.1  # Standard deviation
phi = np.exp(-(x-m)**2/s**2)

# Time integration
for n in range(nt):
    phi_old = phi.copy()
    
    # Periodic boundary conditions for flux calculations
    phi_m1 = np.roll(phi_old, 1)  # i-1 neighbor
    phi_p1 = np.roll(phi_old, -1)  # i+1 neighbor
    
    # Convective fluxes (upwind)
    F_conv = u * np.where(u > 0, phi_old, phi_p1)
    F_conv_m1 = u * np.where(u > 0, phi_m1, phi_old)
    
    # Diffusive fluxes
    F_diff = D * (phi_p1 - phi_old)/dx
    F_diff_m1 = D * (phi_old - phi_m1)/dx
    
    # Update solution
    phi = phi_old - dt/dx * (F_conv - F_conv_m1) + dt/dx * (F_diff - F_diff_m1)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/phi_1D_Unsteady_Convection_Diffusion_Periodic.npy', phi)