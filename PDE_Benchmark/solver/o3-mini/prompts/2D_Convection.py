import numpy as np

# Domain parameters
nx, ny = 101, 101
xmin, xmax = 0.0, 2.0
ymin, ymax = 0.0, 2.0
dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

# Time parameters
t0 = 0.0
t_final = 0.32
CFL = 0.2

# Create grid
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initialize solution arrays for u and v
u = np.ones((nx, ny))
v = np.ones((nx, ny))

# Set initial condition: u=v=2 for 0.5 <= x <= 1 and 0.5 <= y <= 1
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2.0
v[mask] = 2.0

# Time stepping: determine dt from CFL condition based on maximum velocities
dt = CFL * min(dx/np.max(u), dy/np.max(v))
t = t0

# Time loop
while t < t_final:
    # Adjust dt for final time step
    if t + dt > t_final:
        dt = t_final - t
    
    # Create copies for the new step
    u_new = u.copy()
    v_new = v.copy()
    
    # Loop over interior points (using first order upwind scheme; u, v > 0 so use backward difference)
    # i index corresponds to x direction, j corresponds to y direction
    for i in range(1, nx):
        for j in range(1, ny):
            du_dx = (u[i, j] - u[i-1, j]) / dx
            du_dy = (u[i, j] - u[i, j-1]) / dy
            dv_dx = (v[i, j] - v[i-1, j]) / dx
            dv_dy = (v[i, j] - v[i, j-1]) / dy

            u_new[i, j] = u[i, j] - dt*( u[i, j]*du_dx + v[i, j]*du_dy )
            v_new[i, j] = v[i, j] - dt*( u[i, j]*dv_dx + v[i, j]*dv_dy )
            
    # Reapply Dirichlet boundary conditions (u = 1, v = 1 on all boundaries)
    u_new[0, :] = 1.0
    u_new[-1, :] = 1.0
    u_new[:, 0] = 1.0
    u_new[:, -1] = 1.0

    v_new[0, :] = 1.0
    v_new[-1, :] = 1.0
    v_new[:, 0] = 1.0
    v_new[:, -1] = 1.0

    # Update the solution and time
    u = u_new.copy()
    v = v_new.copy()
    t += dt

# Save final solution at t = t_final for both variables as 2D NumPy arrays
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Convection.npy', v)