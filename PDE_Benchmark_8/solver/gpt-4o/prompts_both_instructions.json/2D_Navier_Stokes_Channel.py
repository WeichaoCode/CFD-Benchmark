import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 41, 41
nt = 10
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = 0.01
rho = 1
nu = 0.1
F = 1

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Function to apply boundary conditions
def apply_boundary_conditions(u, v, p):
    # Periodic BCs in x-direction
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]
    v[:, 0] = v[:, -2]
    v[:, -1] = v[:, 1]
    p[:, 0] = p[:, -2]
    p[:, -1] = p[:, 1]
    
    # No-slip BCs in y-direction
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :] = 0
    
    # Neumann BCs for pressure in y-direction
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]

# Function to build the source term for the pressure Poisson equation
def build_up_b(b, u, v, dx, dy, rho, dt):
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + 
                     (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                      ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

# Function to solve the pressure Poisson equation
def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    for _ in range(50):  # Number of iterations for convergence
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])
        apply_boundary_conditions(u, v, p)

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    build_up_b(b, u, v, dx, dy, rho, dt)
    pressure_poisson(p, dx, dy, b)
    
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * 
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy * 
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * 
                    (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx**2 * 
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     dt / dy**2 * 
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + F * dt)
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * 
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy * 
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt / (2 * rho * dy) * 
                    (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt / dx**2 * 
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     dt / dy**2 * 
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
    
    apply_boundary_conditions(u, v, p)

# Save the final solution
np.save('final_solution.npy', {'u': u, 'v': v, 'p': p})

# Visualization
plt.figure(figsize=(11, 7), dpi=100)
plt.quiver(np.linspace(0, 2, nx), np.linspace(0, 2, ny), u, v)
plt.title('Velocity Field')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.figure(figsize=(11, 7), dpi=100)
plt.contourf(np.linspace(0, 2, nx), np.linspace(0, 2, ny), p, alpha=0.5, cmap='viridis')
plt.colorbar()
plt.title('Pressure Field')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()