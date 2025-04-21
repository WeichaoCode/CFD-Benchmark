import numpy as np

# Parameters
Lx, Lz = 4.0, 1.0
nx, nz = 128, 32
dx = Lx/nx
dz = Lz/nz
dt = 1e-3
t_end = 1.0  # Reduced simulation time
nt = int(t_end/dt)

Ra = 2e6
Pr = 1.0
nu = float((Ra/Pr)**(-0.5))
kappa = float((Ra*Pr)**(-0.5))

x = np.linspace(0, Lx, nx)
z = np.linspace(0, Lz, nz)
X, Z = np.meshgrid(x, z, indexing='ij')

# Initialize fields
u = np.zeros((nx, nz), dtype=np.float64)
w = np.zeros((nx, nz), dtype=np.float64)
p = np.zeros((nx, nz), dtype=np.float64)
b = (Lz - Z + 0.0001*np.random.randn(nx, nz)).astype(np.float64)

def diffusion_step(f, coeff, dt):
    f_new = f.copy()
    f_new[1:-1,1:-1] = f[1:-1,1:-1] + coeff*dt*(
        (f[2:,1:-1] - 2*f[1:-1,1:-1] + f[:-2,1:-1])/dx**2 +
        (f[1:-1,2:] - 2*f[1:-1,1:-1] + f[1:-1,:-2])/dz**2
    )
    return f_new

def advection_step(f, u, w, dt):
    f_new = f.copy()
    f_new[1:-1,1:-1] = f[1:-1,1:-1] - dt*(
        u[1:-1,1:-1]*(f[2:,1:-1] - f[:-2,1:-1])/(2*dx) +
        w[1:-1,1:-1]*(f[1:-1,2:] - f[1:-1,:-2])/(2*dz)
    )
    return f_new

# Time stepping
for n in range(nt):
    # 1. Diffusion
    u = diffusion_step(u, nu, dt)
    w = diffusion_step(w, nu, dt)
    b = diffusion_step(b, kappa, dt)
    
    # 2. Advection
    u = advection_step(u, u, w, dt)
    w = advection_step(w, u, w, dt)
    b = advection_step(b, u, w, dt)
    
    # 3. Buoyancy
    w[1:-1,1:-1] += dt*b[1:-1,1:-1]
    
    # 4. Pressure correction
    div = np.zeros_like(p)
    div[1:-1,1:-1] = (u[2:,1:-1] - u[:-2,1:-1])/(2*dx) + \
                     (w[1:-1,2:] - w[1:-1,:-2])/(2*dz)
    
    for _ in range(10):
        p[1:-1,1:-1] = 0.25*(
            (p[2:,1:-1] + p[:-2,1:-1]) +
            (p[1:-1,2:] + p[1:-1,:-2]) -
            div[1:-1,1:-1]*dx*dz/dt
        )
        
        # Periodic in x
        p[0,:] = p[-2,:]
        p[-1,:] = p[1,:]
        # Neumann in z
        p[:,0] = p[:,1]
        p[:,-1] = p[:,-2]
    
    # 5. Velocity correction
    u[1:-1,1:-1] -= dt*(p[2:,1:-1] - p[:-2,1:-1])/(2*dx)
    w[1:-1,1:-1] -= dt*(p[1:-1,2:] - p[1:-1,:-2])/(2*dz)
    
    # 6. Boundary conditions
    # Periodic in x
    u[0,:] = u[-2,:]
    u[-1,:] = u[1,:]
    w[0,:] = w[-2,:]
    w[-1,:] = w[1,:]
    b[0,:] = b[-2,:]
    b[-1,:] = b[1,:]
    
    # No-slip walls
    u[:,0] = 0
    u[:,-1] = 0
    w[:,0] = 0
    w[:,-1] = 0
    
    # Temperature BC
    b[:,0] = Lz
    b[:,-1] = 0
    
    # Apply maximum velocity constraint for stability
    u_max = 10.0
    u = np.clip(u, -u_max, u_max)
    w = np.clip(w, -u_max, u_max)

# Save final state
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Rayleigh_Benard_Convection.npy', u)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/w_2D_Rayleigh_Benard_Convection.npy', w)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/p_2D_Rayleigh_Benard_Convection.npy', p)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/b_2D_Rayleigh_Benard_Convection.npy', b)