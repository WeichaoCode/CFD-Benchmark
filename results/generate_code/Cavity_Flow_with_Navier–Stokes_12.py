import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx = 101
ny = 101
nt = 500
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
dt = 2.0/nt

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Physical parameters
rho = 1.0
nu = 0.1

# Initialize variables
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

un = np.zeros((ny, nx))
vn = np.zeros((ny, nx))
pn = np.zeros((ny, nx))

def build_up_b(u, v, dx, dy, dt, rho):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1/dt * ((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx) + 
                                    (v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx))**2 -
                            2*((u[2:, 1:-1] - u[0:-2, 1:-1])/(2*dy) *
                               (v[1:-1, 2:] - v[1:-1, 0:-2])/(2*dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))**2))
    return b

def pressure_poisson(p, b, dx, dy):
    pn = np.zeros_like(p)
    
    for q in range(50):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2])*dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1])*dx**2)/(2*(dx**2 + dy**2)) -
                          dx**2*dy**2/(2*(dx**2 + dy**2))*b[1:-1, 1:-1])
        
        # Boundary conditions
        p[:, -1] = p[:, -2]    # dp/dx = 0 at x = 2
        p[:, 0] = p[:, 1]      # dp/dx = 0 at x = 0
        p[0, :] = p[1, :]      # dp/dy = 0 at y = 0
        p[-1, :] = 0           # p = 0 at y = 2
        
    return p

# Time stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    b = build_up_b(u, v, dx, dy, dt, rho)
    p = pressure_poisson(p, b, dx, dy)
    
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                     un[1:-1, 1:-1]*dt/dx*(un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1]*dt/dy*(un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt/(2*rho*dx)*(p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu*(dt/dx**2*(un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt/dy**2*(un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[0:-2, 1:-1])))
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1]*dt/dx*(vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1]*dt/dy*(vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt/(2*rho*dy)*(p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu*(dt/dx**2*(vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                         dt/dy**2*(vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
    
    # Boundary conditions
    u[0, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    u[-1, :] = 1    # Moving lid
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0

# Plot results
plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, u, v)
plt.title('Cavity Flow - Streamlines')
plt.xlabel('x')
plt.ylabel('y')
plt.show()