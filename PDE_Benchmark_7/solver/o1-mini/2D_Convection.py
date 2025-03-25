import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define Parameters
nx = ny = 101
nt = 80
sigma = 0.2
Lx = Ly = 2.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = sigma * dx

# 2. Initialize Variables
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set initial condition: 2 in the region 0.5 <= x,y <=1
u_initial_region = np.where((X >= 0.5) & (X <=1.0) & (Y >=0.5) & (Y <=1.0))
v_initial_region = np.where((X >= 0.5) & (X <=1.0) & (Y >=0.5) & (Y <=1.0))
u[u_initial_region] = 2.0
v[v_initial_region] = 2.0

# 3. Time Integration Loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Apply boundary conditions
    un[0, :] = un[-1, :] = 1
    un[:, 0] = un[:, -1] = 1
    vn[0, :] = vn[-1, :] = 1
    vn[:, 0] = vn[:, -1] = 1
    
    # Compute interior points
    u[1:-1,1:-1] = (un[1:-1,1:-1] -
                    dt * un[1:-1,1:-1] *
                    ((un[1:-1,1:-1] - un[1:-1,0:-2]) / dx +
                     (vn[1:-1,1:-1] - vn[0:-2,1:-1]) / dy))
    
    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                    dt * vn[1:-1,1:-1] *
                    ((un[1:-1,1:-1] - un[1:-1,0:-2]) / dx +
                     (vn[1:-1,1:-1] - vn[0:-2,1:-1]) / dy))

# Apply boundary conditions at final step
u[0, :] = u[-1, :] = 1
u[:, 0] = u[:, -1] = 1
v[0, :] = v[-1, :] = 1
v[:, 0] = v[:, -1] = 1

# 4. Visualization
fig = plt.figure(figsize=(14,6))

# Surface plot for u
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, u, cmap='viridis')
ax1.set_title('Velocity Field u at Final Time Step')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('u')

# Surface plot for v
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, v, cmap='viridis')
ax2.set_title('Velocity Field v at Final Time Step')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('v')

plt.tight_layout()
plt.show()

# Save the final velocity fields
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_2D_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/v_2D_Convection.npy', v)