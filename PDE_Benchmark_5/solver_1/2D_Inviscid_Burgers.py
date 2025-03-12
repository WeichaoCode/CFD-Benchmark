import numpy as np
import matplotlib.pyplot as plt

# 1. Define parameters 
Lx, Ly, T = 2.0, 2.0, 1.0  # lengths and maximum time
nx, ny, nt = 101, 101, 101  # number of points in x, y and t
dx, dy, dt = Lx / (nx - 1), Ly / (ny - 1), T / (nt - 1)  # step size 

# 2. Discretize space and time
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.linspace(0, T, nt)

# 3. Setup initial velocity field
u = np.zeros((ny, nx))  # u-velocity
v = np.zeros((ny, nx))  # v-velocity
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2  # initial condition
v[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2  # initial condition

# 4. Iterate using finite difference
for n in range(nt - 1):
    un = u.copy()
    vn = v.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - 
                     vn[1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]))
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt/dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]))

# 5. Visualization
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(figsize=(7,7))
q = ax.quiver(Y[::3, ::3], X[::3, ::3], u[::3, ::3], v[::3, ::3])
plt.show()