import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nx = 101
ny = 101
nt = 80
Lx = 2.0
Ly = 2.0
sigma = 0.2

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = sigma * dx

# Initialize velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial condition: u = v = 2 for 0.5 <= x, y <= 1
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

u[(X >= 0.5) & (X <= 1) & (Y >= 0.5) & (Y <= 1)] = 2
v[(X >= 0.5) & (X <= 1) & (Y >= 0.5) & (Y <= 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Upwind scheme for u
    u[1:, 1:] = (un[1:, 1:] - 
                 dt / dx * un[1:, 1:] * (un[1:, 1:] - un[1:, :-1]) - 
                 dt / dy * vn[1:, 1:] * (un[1:, 1:] - un[:-1, 1:]))
    
    # Upwind scheme for v
    v[1:, 1:] = (vn[1:, 1:] - 
                 dt / dx * un[1:, 1:] * (vn[1:, 1:] - vn[1:, :-1]) - 
                 dt / dy * vn[1:, 1:] * (vn[1:, 1:] - vn[:-1, 1:]))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Save final solution to .npy files
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_2D_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/v_2D_Convection.npy', v)

# Visualization
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax1.plot_surface(X, Y, u, cmap='viridis')
ax1.set_title('Final u')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2.plot_surface(X, Y, v, cmap='viridis')
ax2.set_title('Final v')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.tight_layout()
plt.show()