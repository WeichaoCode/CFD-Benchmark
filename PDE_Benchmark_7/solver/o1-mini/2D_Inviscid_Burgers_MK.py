import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 2.0, 2.0
nx, ny = 151, 151
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
nt = 300
sigma = 0.2
dt = sigma * min(dx, dy) / 2

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initialize velocity fields
u = np.ones((nx, ny))
v = np.ones((nx, ny))

# Apply initial condition: u and v = 2 in the region 0.5 <= x <=1 and 0.5 <= y <=1
condition = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[condition] = 2.0
v[condition] = 2.0

# Time-stepping loop
for n in range(nt):
    # Create copies of u and v to hold previous time step values
    u_prev = u.copy()
    v_prev = v.copy()
    
    # Update interior points using First-Order Upwind Scheme
    u[1:,1:] = u_prev[1:,1:] - dt * (
        u_prev[1:,1:] * (u_prev[1:,1:] - u_prev[:-1,1:]) / dx +
        v_prev[1:,1:] * (u_prev[1:,1:] - u_prev[1:,:-1]) / dy
    )
    
    v[1:,1:] = v_prev[1:,1:] - dt * (
        u_prev[1:,1:] * (v_prev[1:,1:] - v_prev[:-1,1:]) / dx +
        v_prev[1:,1:] * (v_prev[1:,1:] - v_prev[1:,:-1]) / dy
    )
    
    # Apply Dirichlet boundary conditions: u = 1, v = 1 on all boundaries
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Visualization using quiver plot
plt.figure(figsize=(8, 6))
skip = 5  # to reduce the number of arrows in the plot
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
           u[::skip, ::skip], v[::skip, ::skip],
           pivot='mid', color='r')
plt.title('Velocity Field at Final Time Step')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.grid()
plt.show()

# Save the final velocity fields
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_2D_Inviscid_Burgers_MK.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/v_2D_Inviscid_Burgers_MK.npy', v)