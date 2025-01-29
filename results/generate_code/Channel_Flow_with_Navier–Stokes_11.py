import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
Nx = 101
Ny = 101
Lx = 2.0
Ly = 2.0
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Time parameters
T = 2.0
nt = 500
dt = T/nt

# Physical parameters
rho = 1.0
nu = 0.1
F = 1.0

# Initialize fields
u = np.zeros((Ny, Nx))
v = np.zeros((Ny, Nx))
p = np.zeros((Ny, Nx))
un = np.zeros((Ny, Nx))
vn = np.zeros((Ny, Nx))
pn = np.zeros((Ny, Nx))

def periodic_bc(f):
    f[:,0] = f[:,-2]
    f[:,-1] = f[:,1]
    return f

def solve_pressure_poisson(p, u, v, dx, dy, rho):
    pn = np.zeros_like(p)
    
    for it in range(50):
        pn = p.copy()
        p[1:-1,1:-1] = 0.25*(pn[1:-1,2:] + pn[1:-1,:-2] + pn[2:,1:-1] + pn[:-2,1:-1] - 
                            rho*dx*dy*((u[1:-1,2:] - u[1:-1,:-2])/(2*dx) * (u[1:-1,2:] - u[1:-1,:-2])/(2*dx) +
                                     2*(u[2:,1:-1] - u[:-2,1:-1])/(2*dy) * (v[1:-1,2:] - v[1:-1,:-2])/(2*dx) +
                                     (v[2:,1:-1] - v[:-2,1:-1])/(2*dy) * (v[2:,1:-1] - v[:-2,1:-1])/(2*dy)))
        
        # Periodic BC in x
        p[:,0] = p[:,-2]
        p[:,-1] = p[:,1]
        
        # Neumann BC in y
        p[0,:] = p[1,:]
        p[-1,:] = p[-2,:]
        
    return p

# Time stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # X-momentum
    u[1:-1,1:-1] = un[1:-1,1:-1] - \
                   dt*(un[1:-1,1:-1]*(un[1:-1,2:] - un[1:-1,:-2])/(2*dx) +
                       vn[1:-1,1:-1]*(un[2:,1:-1] - un[:-2,1:-1])/(2*dy)) - \
                   dt/(rho)*(p[1:-1,2:] - p[1:-1,:-2])/(2*dx) + \
                   nu*dt*((un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])/dx**2 +
                         (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1])/dy**2) + \
                   F*dt
    
    # Y-momentum
    v[1:-1,1:-1] = vn[1:-1,1:-1] - \
                   dt*(un[1:-1,1:-1]*(vn[1:-1,2:] - vn[1:-1,:-2])/(2*dx) +
                       vn[1:-1,1:-1]*(vn[2:,1:-1] - vn[:-2,1:-1])/(2*dy)) - \
                   dt/(rho)*(p[2:,1:-1] - p[:-2,1:-1])/(2*dy) + \
                   nu*dt*((vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])/dx**2 +
                         (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1])/dy**2)
    
    # Periodic BC in x
    u = periodic_bc(u)
    v = periodic_bc(v)
    
    # No-slip BC in y
    u[0,:] = 0
    u[-1,:] = 0
    v[0,:] = 0
    v[-1,:] = 0
    
    # Pressure Poisson equation
    p = solve_pressure_poisson(p, u, v, dx, dy, rho)

# Plot final results
plt.figure(figsize=(10,5))

plt.subplot(121)
plt.contourf(X, Y, u, levels=np.linspace(u.min(), u.max(), 50))
plt.colorbar(label='u')
plt.title('u velocity')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(122)
plt.contourf(X, Y, v, levels=np.linspace(v.min(), v.max(), 50))
plt.colorbar(label='v')
plt.title('v velocity')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()