import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx = 101
ny = 101
dx = 2.0 / (nx-1)
dy = 2.0 / (ny-1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Physical parameters
rho = 1.0
nu = 0.1
dt = 2.0 / 500

# Initialize variables
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

def build_up_b(u, v):
    b = np.zeros((ny, nx))
    b[1:-1,1:-1] = (rho * (1/dt * ((u[1:-1,2:] - u[1:-1,0:-2])/(2*dx) + 
                                   (v[2:,1:-1] - v[0:-2,1:-1])/(2*dy)) -
                           ((u[1:-1,2:] - u[1:-1,0:-2])/(2*dx))**2 -
                           2*((u[2:,1:-1] - u[0:-2,1:-1])/(2*dy) *
                              (v[1:-1,2:] - v[1:-1,0:-2])/(2*dx)) -
                           ((v[2:,1:-1] - v[0:-2,1:-1])/(2*dy))**2))
    return b

def pressure_poisson(p, b):
    pn = np.empty_like(p)
    for q in range(50):
        pn = p.copy()
        p[1:-1,1:-1] = ((pn[1:-1,2:] + pn[1:-1,0:-2])*dy**2 + 
                        (pn[2:,1:-1] + pn[0:-2,1:-1])*dx**2) / (2*(dx**2 + dy**2)) - \
                        dx**2*dy**2/(2*(dx**2 + dy**2))*b[1:-1,1:-1]
        
        # Boundary conditions
        p[-1,:] = 0  # p = 0 at y = 2
        p[0,:] = p[1,:]  # dp/dy = 0 at y = 0
        p[:,0] = p[:,1]  # dp/dx = 0 at x = 0
        p[:,-1] = p[:,-2]  # dp/dx = 0 at x = 2
        
    return p

# Time stepping
for n in range(500):
    un = u.copy()
    vn = v.copy()
    
    # u-momentum equation
    u[1:-1,1:-1] = un[1:-1,1:-1] - \
                   dt/dx*un[1:-1,1:-1]*(un[1:-1,1:-1] - un[1:-1,0:-2]) - \
                   dt/dy*vn[1:-1,1:-1]*(un[1:-1,1:-1] - un[0:-2,1:-1]) - \
                   dt/(2*rho*dx)*(p[1:-1,2:] - p[1:-1,0:-2]) + \
                   nu*dt/dx**2*(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]) + \
                   nu*dt/dy**2*(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1])
    
    # v-momentum equation
    v[1:-1,1:-1] = vn[1:-1,1:-1] - \
                   dt/dx*un[1:-1,1:-1]*(vn[1:-1,1:-1] - vn[1:-1,0:-2]) - \
                   dt/dy*vn[1:-1,1:-1]*(vn[1:-1,1:-1] - vn[0:-2,1:-1]) - \
                   dt/(2*rho*dy)*(p[2:,1:-1] - p[0:-2,1:-1]) + \
                   nu*dt/dx**2*(vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,0:-2]) + \
                   nu*dt/dy**2*(vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[0:-2,1:-1])
    
    # Boundary conditions
    u[-1,:] = 1    # u = 1 at y = 2 (lid)
    u[0,:] = 0     # u = 0 at y = 0
    u[:,0] = 0     # u = 0 at x = 0
    u[:,-1] = 0    # u = 0 at x = 2
    
    v[-1,:] = 0    # v = 0 at y = 2
    v[0,:] = 0     # v = 0 at y = 0
    v[:,0] = 0     # v = 0 at x = 0
    v[:,-1] = 0    # v = 0 at x = 2
    
    # Pressure correction
    b = build_up_b(u, v)
    p = pressure_poisson(p, b)

# Plot results
plt.figure(figsize=(10,10))
plt.streamplot(X, Y, u, v)
plt.title('Cavity Flow')
plt.xlabel('x')
plt.ylabel('y')
plt.show()