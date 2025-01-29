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

# Initialize variables
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
    pn = p.copy()
    for it in range(50):
        p[1:-1,1:-1] = 0.25*(pn[1:-1,2:] + pn[1:-1,:-2] + pn[2:,1:-1] + pn[:-2,1:-1] - 
                             rho*dx*dy*((u[1:-1,2:] - u[1:-1,:-2])/(2*dx) * (u[1:-1,2:] - u[1:-1,:-2])/(2*dx) +
                                      2*(u[2:,1:-1] - u[:-2,1:-1])/(2*dy) * (v[1:-1,2:] - v[1:-1,:-2])/(2*dx) +
                                      (v[2:,1:-1] - v[:-2,1:-1])/(2*dy) * (v[2:,1:-1] - v[:-2,1:-1])/(2*dy)))
        p = periodic_bc(p)
        p[0,:] = p[1,:]
        p[-1,:] = p[-2,:]
    return p

# Time stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Solve momentum equations
    u[1:-1,1:-1] = un[1:-1,1:-1] - dt*(
        un[1:-1,1:-1]*(un[1:-1,2:] - un[1:-1,:-2])/(2*dx) +
        vn[1:-1,1:-1]*(un[2:,1:-1] - un[:-2,1:-1])/(2*dy)
    ) - dt/(rho)*(p[1:-1,2:] - p[1:-1,:-2])/(2*dx) + dt*nu*(
        (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])/dx**2 +
        (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1])/dy**2
    ) + dt*F
    
    v[1:-1,1:-1] = vn[1:-1,1:-1] - dt*(
        un[1:-1,1:-1]*(vn[1:-1,2:] - vn[1:-1,:-2])/(2*dx) +
        vn[1:-1,1:-1]*(vn[2:,1:-1] - vn[:-2,1:-1])/(2*dy)
    ) - dt/(rho)*(p[2:,1:-1] - p[:-2,1:-1])/(2*dy) + dt*nu*(
        (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])/dx**2 +
        (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1])/dy**2
    )
    
    # Apply boundary conditions
    u = periodic_bc(u)
    v = periodic_bc(v)
    u[0,:] = 0
    u[-1,:] = 0
    v[0,:] = 0
    v[-1,:] = 0
    
    # Solve pressure Poisson equation
    p = solve_pressure_poisson(p, u, v, dx, dy, rho)

# Plot results
plt.figure(figsize=(10,5))

plt.subplot(121)
plt.contourf(X, Y, u, levels=np.linspace(u.min(), u.max(), 50))
plt.colorbar(label='u velocity')
plt.title('u velocity')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(122)
plt.contourf(X, Y, v, levels=np.linspace(v.min(), v.max(), 50))
plt.colorbar(label='v velocity')
plt.title('v velocity')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()