import numpy as np

# Parameters
nu = 0.07
L = 2*np.pi
T = 0.14*np.pi
nx = 400
nt = 1000
dx = L/nx
dt = T/nt
x = np.linspace(0, L, nx)

# Initial condition
def phi(x):
    return np.exp(-x**2/(4*nu)) + np.exp(-(x-2*np.pi)**2/(4*nu))

def dphi_dx(x):
    return -x/(2*nu)*np.exp(-x**2/(4*nu)) + \
           -(x-2*np.pi)/(2*nu)*np.exp(-(x-2*np.pi)**2/(4*nu))

# Initialize u
u = -2*nu/phi(x) * dphi_dx(x) + 4

# Time stepping
for n in range(nt):
    # Periodic BC handled through array indexing
    un = u.copy()
    
    # Space derivatives
    du_dx = np.zeros_like(u)
    d2u_dx2 = np.zeros_like(u)
    
    # Central difference for diffusion
    d2u_dx2[1:-1] = (un[2:] - 2*un[1:-1] + un[:-2])/dx**2
    d2u_dx2[0] = (un[1] - 2*un[0] + un[-1])/dx**2
    d2u_dx2[-1] = d2u_dx2[0]
    
    # Upwind difference for convection
    for i in range(nx):
        if un[i] > 0:
            if i == 0:
                du_dx[i] = (un[i] - un[-1])/dx
            else:
                du_dx[i] = (un[i] - un[i-1])/dx
        else:
            if i == nx-1:
                du_dx[i] = (un[0] - un[i])/dx
            else:
                du_dx[i] = (un[i+1] - un[i])/dx
    
    # Update
    u = un - dt*(un*du_dx - nu*d2u_dx2)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_1D_Burgers_Equation.npy', u)