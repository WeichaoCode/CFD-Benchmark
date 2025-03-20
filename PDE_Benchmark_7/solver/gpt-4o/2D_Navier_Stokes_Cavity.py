import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 51, 51                       # Number of grid points in x and y
nt = 500                              # Number of time steps
dt = 0.001                            # Time step size
Lx, Ly = 2.0, 2.0                     # Domain length in x and y
dx, dy = Lx / (nx - 1), Ly / (ny - 1) # Grid spacing
rho = 1.0                             # Density
nu = 0.1                              # Kinematic viscosity

# Initialize fields
u = np.zeros((ny, nx))                # Velocity field in x direction
v = np.zeros((ny, nx))                # Velocity field in y direction
p = np.zeros((ny, nx))                # Pressure field

# Discrete Laplace operator
def laplace(phi):
    return (phi[2:, 1:-1] + phi[:-2, 1:-1] - 2 * phi[1:-1, 1:-1]) / dx**2 + \
           (phi[1:-1, 2:] + phi[1:-1, :-2] - 2 * phi[1:-1, 1:-1]) / dy**2

# Time-stepping
for n in range(nt):
    un, vn, pn = u.copy(), v.copy(), p.copy()
    
    # Convective terms
    conv_u = un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]) / dx + \
             vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1]) / dy
             
    conv_v = un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) / dx + \
             vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) / dy
    
    # Momentum equations (Navier-Stokes)
    u[1:-1, 1:-1] = un[1:-1, 1:-1] + dt * (
        - conv_u - (pn[2:, 1:-1] - pn[:-2, 1:-1]) / (2 * dx * rho) +
        nu * laplace(un)
    )
    
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt * (
        - conv_v - (pn[1:-1, 2:] - pn[1:-1, :-2]) / (2 * dy * rho) +
        nu * laplace(vn)
    )

    # Pressure Poisson equation
    rhs = rho * ((un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dx))**2 + \
          2 * (((un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dy)) *
               ((vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dx))) + \
          ((vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dy))**2
           
    for _ in range(50): # Iterative solver for the Poisson equation
        pn[1:-1, 1:-1] = ((pn[2:, 1:-1] + pn[:-2, 1:-1]) * dy**2 + 
                          (pn[1:-1, 2:] + pn[1:-1, :-2]) * dx**2) / (2 * (dx**2 + dy**2)) + \
                         (rhs * dx**2 * dy**2) / (2 * (dx**2 + dy**2))
                          
        # Apply boundary conditions for pressure
        pn[:, 0] = pn[:, 1]               # dp/dx = 0 at x = 0 (left)
        pn[:, -1] = pn[:, -2]             # dp/dx = 0 at x = Lx (right)
        pn[0, :] = pn[1, :]               # dp/dy = 0 at y = 0 (bottom)
        pn[-1, :] = 0                     # p = 0 at y = Ly (top)
    
    p = pn.copy()

    # Boundary conditions for velocity
    u[0, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    u[-1, :] = 1     # Lid driven condition at the top boundary

    v[0, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0
    v[-1, :] = 0

# Visualization
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

plt.figure(figsize=(12, 5))

# Velocity field (quiver plot)
plt.subplot(1, 2, 1)
plt.quiver(X, Y, u, v)
plt.title('Velocity field')
plt.xlabel('X')
plt.ylabel('Y')

# Pressure distribution (contour plot)
plt.subplot(1, 2, 2)
plt.contourf(X, Y, p, alpha=0.7, cmap='viridis')
plt.colorbar()
plt.title('Pressure distribution')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()

# Save results
np.save('velocity_u.npy', u)
np.save('velocity_v.npy', v)
np.save('pressure.npy', p)