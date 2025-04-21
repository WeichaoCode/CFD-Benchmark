import numpy as np

# Parameters
c = 1.0              # convection speed
x_start, x_end = 0, 2
y_start, y_end = 0, 2
T = 0.50             # final time

# Grid parameters
nx = 101             # number of grid points in x
ny = 101             # number of grid points in y
dx = (x_end - x_start) / (nx - 1)
dy = (y_end - y_start) / (ny - 1)

# CFL condition for stability (using an upwind scheme)
CFL = 0.5
dt = CFL * min(dx, dy) / c

# Create grid
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initialize u with Dirichlet boundaries u=1 everywhere
u = np.ones((nx, ny))

# Set initial conditions: u = 2 for 0.5 <= x <= 1 and 0.5 <= y <= 1 (inside the domain)
inside = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[inside] = 2.0

# Enforce boundary conditions explicitly (though u is already 1 at boundaries)
u[0, :] = 1.0
u[-1, :] = 1.0
u[:, 0] = 1.0
u[:, -1] = 1.0

# Time stepping loop
t = 0.0
while t < T:
    # Make a copy for new time step
    u_new = u.copy()
    
    # Update interior points using an explicit upwind scheme,
    # since c>0 we use backward differences:
    # u^{n+1}[i,j] = u^n[i,j] - c*dt/dx*(u[i,j] - u[i-1,j]) - c*dt/dy*(u[i,j] - u[i,j-1])
    u_new[1:, 1:] = (u[1:, 1:] -
                     c * dt/dx * (u[1:, 1:] - u[:-1, 1:]) -
                     c * dt/dy * (u[1:, 1:] - u[1:, :-1]))
    
    # Re-enforce Dirichlet boundary conditions: u = 1 on boundaries
    u_new[0, :] = 1.0
    u_new[-1, :] = 1.0
    u_new[:, 0] = 1.0
    u_new[:, -1] = 1.0
    
    # Update u and time
    u = u_new.copy()
    t += dt
    if t + dt > T:
        dt = T - t  # adjust last time step if needed

# Save the final solution as specified (a 2D numpy array saved in 'u.npy')
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Linear_Convection.npy', u)