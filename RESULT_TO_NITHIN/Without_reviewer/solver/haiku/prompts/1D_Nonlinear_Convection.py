import numpy as np

# Domain setup
nx = 200  # Number of spatial points
nt = 1000  # Number of time steps
x = np.linspace(0, 2*np.pi, nx)
dx = x[1] - x[0]
t = np.linspace(0, 5, nt)
dt = t[1] - t[0]

# Initial condition
u = np.sin(x) + 0.5*np.sin(0.5*x)

# Time stepping with Lax-Wendroff method
for n in range(nt-1):
    # Predictor step (half time step)
    u_half = np.zeros_like(u)
    for i in range(nx-1):
        u_half[i] = 0.5*(u[i] + u[i+1]) - 0.25*dt/dx*u[i]*(u[i+1] - u[i])
    u_half[-1] = u_half[0]  # Periodic BC
    
    # Corrector step
    u_new = np.zeros_like(u)
    for i in range(1,nx):
        u_new[i] = u[i] - dt/dx*u_half[i]*(u_half[i] - u_half[i-1])
    u_new[0] = u_new[-1]  # Periodic BC
    
    u = u_new.copy()

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_1D_Nonlinear_Convection.npy', u)