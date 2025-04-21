import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags

# Grid parameters
Nx = 128
Nz = 128
dx = 1.0/Nx
dz = 2.0/Nz
x = np.linspace(0, 1, Nx)
z = np.linspace(-1, 1, Nz)
X, Z = np.meshgrid(x, z)

# Time parameters
dt = 0.001
t_final = 20
Nt = int(t_final/dt)

# Physical parameters
nu = 1/(5e4)
D = nu/1

# Initial conditions
u = 0.5*(1 + np.tanh((Z-0.5)/0.1) - np.tanh((Z+0.5)/0.1))
w = 0.01*np.sin(2*np.pi*X)*np.exp(-(Z-0.5)**2/0.1**2) + \
    0.01*np.sin(2*np.pi*X)*np.exp(-(Z+0.5)**2/0.1**2)
s = u.copy()
p = np.zeros_like(u)

# Helper functions
def periodic_bc(f):
    f[0,:] = f[-2,:]
    f[-1,:] = f[1,:]
    f[:,0] = f[:,-2]
    f[:,-1] = f[:,1]
    return f

def solve_pressure_poisson(u, w, dx, dz, dt):
    b = np.zeros_like(p)
    for i in range(1,Nx-1):
        for j in range(1,Nz-1):
            b[j,i] = (1/dt)*((u[j,i+1]-u[j,i-1])/(2*dx) + 
                            (w[j+1,i]-w[j-1,i])/(2*dz))
            
    # Build sparse matrix for Poisson equation
    n = (Nx-2)*(Nz-2)
    A = diags([1, 1, -4, 1, 1], [-n+2, -1, 0, 1, n-2], shape=(n,n))
    
    # Solve system
    p_flat = spsolve(A, b[1:-1,1:-1].flatten())
    p[1:-1,1:-1] = p_flat.reshape((Nz-2,Nx-2))
    p = periodic_bc(p)
    return p

# Time stepping
for n in range(Nt):
    # Compute derivatives
    u_x = (np.roll(u,-1,axis=1) - np.roll(u,1,axis=1))/(2*dx)
    u_z = (np.roll(u,-1,axis=0) - np.roll(u,1,axis=0))/(2*dz)
    w_x = (np.roll(w,-1,axis=1) - np.roll(w,1,axis=1))/(2*dx)
    w_z = (np.roll(w,-1,axis=0) - np.roll(w,1,axis=0))/(2*dz)
    s_x = (np.roll(s,-1,axis=1) - np.roll(s,1,axis=1))/(2*dx)
    s_z = (np.roll(s,-1,axis=0) - np.roll(s,1,axis=0))/(2*dz)
    
    u_xx = (np.roll(u,-1,axis=1) - 2*u + np.roll(u,1,axis=1))/dx**2
    u_zz = (np.roll(u,-1,axis=0) - 2*u + np.roll(u,1,axis=0))/dz**2
    w_xx = (np.roll(w,-1,axis=1) - 2*w + np.roll(w,1,axis=1))/dx**2
    w_zz = (np.roll(w,-1,axis=0) - 2*w + np.roll(w,1,axis=0))/dz**2
    s_xx = (np.roll(s,-1,axis=1) - 2*s + np.roll(s,1,axis=1))/dx**2
    s_zz = (np.roll(s,-1,axis=0) - 2*s + np.roll(s,1,axis=0))/dz**2
    
    # Solve pressure
    p = solve_pressure_poisson(u, w, dx, dz, dt)
    p_x = (np.roll(p,-1,axis=1) - np.roll(p,1,axis=1))/(2*dx)
    p_z = (np.roll(p,-1,axis=0) - np.roll(p,1,axis=0))/(2*dz)
    
    # Update velocities and tracer
    u = u + dt*(-u*u_x - w*u_z - p_x + nu*(u_xx + u_zz))
    w = w + dt*(-u*w_x - w*w_z - p_z + nu*(w_xx + w_zz))
    s = s + dt*(-u*s_x - w*s_z + D*(s_xx + s_zz))
    
    # Apply boundary conditions
    u = periodic_bc(u)
    w = periodic_bc(w)
    s = periodic_bc(s)

# Save final state
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/p_2D_Shear_Flow_With_Tracer.npy', p)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)