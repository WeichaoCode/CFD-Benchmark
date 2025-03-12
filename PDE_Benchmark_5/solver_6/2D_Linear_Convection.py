import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

#parameters
Lx = Ly = 2.0   
nx = ny = 101 
dx = dy = Lx / (nx - 1) 
c = 1.0  
dt = 0.0025 
nt = 401  

#grid points
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

#initialize u
u = np.ones((ny, nx))
mask = np.where(np.logical_and(X >= 0.5, X <= 1.0) & np.logical_and(Y >= 0.5, Y <= 1.0))
u[mask] = 2.0 

#advance solution in time
for it in range(nt):
    un = u.copy()
    u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) - 
                             (c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))
    
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

#Visualization
fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30,225)
ax.plot_surface(X, Y, u[:], cmap='viridis')
plt.show()