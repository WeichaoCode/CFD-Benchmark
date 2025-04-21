import numpy as np

# Grid parameters
nx = 200  # Number of spatial points
nt = 1000  # Number of time steps
dx = 10.0/(nx-1)  # Spatial step size
dt = 0.01  # Time step size
x = np.linspace(-5, 5, nx)

# Parameters
c = 1.0  # Convection speed
epsilon = 5e-4  # Damping factor

# Initial condition
u = np.exp(-x**2)

# Time stepping
for n in range(nt):
    # Periodic boundary conditions handled through array indexing
    
    # Spatial derivatives using central differences
    dudx = (np.roll(u,-1) - np.roll(u,1))/(2*dx)
    d2udx2 = (np.roll(u,-1) - 2*u + np.roll(u,1))/dx**2
    
    # Update solution
    u = u - dt*(c*dudx - epsilon*d2udx2)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_1D_Linear_Convection.npy', u)