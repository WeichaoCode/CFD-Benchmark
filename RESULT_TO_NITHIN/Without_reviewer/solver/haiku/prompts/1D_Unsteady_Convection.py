import numpy as np

# Domain parameters
L = 2.0  # Length of domain
nx = 200  # Number of cells
dx = L/nx  # Cell size
x = np.linspace(dx/2, L-dx/2, nx)  # Cell centers
dt = 0.001  # Time step
t_final = 2.5
nt = int(t_final/dt)  # Number of time steps

# Parameters
u = 0.2  # Velocity
m = 0.5  # Mean of initial Gaussian
s = 0.1  # Standard deviation of initial Gaussian

# Initialize solution array
phi = np.exp(-(x - m)**2/s**2)  # Initial condition
phi[0] = 0  # Left boundary
phi[-1] = 0  # Right boundary

# Time stepping loop
for n in range(nt):
    phi_old = phi.copy()
    
    # Update interior points using upwind scheme
    for i in range(1, nx-1):
        if u > 0:
            phi[i] = phi_old[i] - u*dt/dx*(phi_old[i] - phi_old[i-1])
        else:
            phi[i] = phi_old[i] - u*dt/dx*(phi_old[i+1] - phi_old[i])
            
    # Apply boundary conditions
    phi[0] = 0
    phi[-1] = 0

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/phi_1D_Unsteady_Convection.npy', phi)