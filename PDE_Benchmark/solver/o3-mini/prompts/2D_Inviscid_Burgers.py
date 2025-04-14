#!/usr/bin/env python3
import numpy as np

# Domain parameters
nx = 101
ny = 101
xmin, xmax = 0.0, 2.0
ymin, ymax = 0.0, 2.0
dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

# Time stepping parameters
t_final = 0.40
CFL = 0.4  # chosen CFL number
# Since max velocities are 2 in the interior (and 1 on boundaries), we use max = 2 for dt computation.
dt = CFL * min(dx, dy) / 2.0
nt = int(t_final / dt)

# Create spatial grid (not used for update but can be useful)
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

# Initialize u and v fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set the initial condition: u = v = 2 for 0.5 <= x <= 1 and 0.5 <= y <= 1
# Since our array indices: rows correspond to y and cols to x.
X, Y = np.meshgrid(x, y)
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2.0
v[mask] = 2.0

# Enforce Dirichlet boundary conditions at initial time (u, v = 1 on boundary)
u[0, :] = 1.0
u[-1, :] = 1.0
u[:, 0] = 1.0
u[:, -1] = 1.0

v[0, :] = 1.0
v[-1, :] = 1.0
v[:, 0] = 1.0
v[:, -1] = 1.0

# Time integration using explicit upwind scheme (backward difference)
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update for interior nodes: i from 1 to nx-1, j from 1 to ny-1
    # Use backward differences in x (axis=1) and y (axis=0)
    u[1:, 1:] = un[1:, 1:] - dt * (un[1:, 1:] * (un[1:, 1:] - un[1:, :-1]) / dx 
                                 + vn[1:, 1:] * (un[1:, 1:] - un[:-1, 1:]) / dy)
    
    v[1:, 1:] = vn[1:, 1:] - dt * (un[1:, 1:] * (vn[1:, 1:] - vn[1:, :-1]) / dx 
                                 + vn[1:, 1:] * (vn[1:, 1:] - vn[:-1, 1:]) / dy)
    
    # Re-apply Dirichlet boundary conditions at all boundaries
    u[0, :] = 1.0
    u[-1, :] = 1.0
    u[:, 0] = 1.0
    u[:, -1] = 1.0

    v[0, :] = 1.0
    v[-1, :] = 1.0
    v[:, 0] = 1.0
    v[:, -1] = 1.0

# Save the final solution fields
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Inviscid_Burgers.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Inviscid_Burgers.npy', v)