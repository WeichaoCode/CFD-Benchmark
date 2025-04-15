import numpy as np

# Grid parameters
nx = 200  # Number of spatial points
nt = 1000  # Number of time steps
dx = 2*np.pi/nx
dt = 0.001  # Reduced time step for stability
x = np.linspace(0, 2*np.pi, nx)
t = np.linspace(0, 5, nt)

# Initial condition
u = np.sin(x) + 0.5*np.sin(0.5*x)

# Time integration using upwind scheme
for n in range(nt-1):
    u_new = u.copy()
    
    for i in range(nx):
        im1 = (i-1) % nx  # Periodic BC
        
        # First-order upwind scheme
        if u[i] > 0:
            u_new[i] = u[i] - dt*u[i]*(u[i] - u[im1])/dx
        else:
            ip1 = (i+1) % nx
            u_new[i] = u[i] - dt*u[i]*(u[ip1] - u[i])/dx
    
    u = u_new.copy()

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_1D_Nonlinear_Convection.npy', u)