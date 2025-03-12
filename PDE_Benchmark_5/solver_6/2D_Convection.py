import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
Lx, Ly = 2, 2  # domain size
T = 2           # total time
nx, ny = 101, 101  # number of grid points
nt = 100  # number of time steps

dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # grid increments
dt = T / nt  # time increment

cfl = dt * (1/dx + 1/dy)  # CFL number

# Step 2: Discretize space and time.
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

u = np.ones((ny, nx))  # Initialize u
v = np.ones((ny, nx))  # Initialize v

# Step 3: Apply the initial condition.
mask = np.logical_and(x > 0.5, x < 1)
u[:, mask], v[:, mask] = 2, 2

for n in range(nt):  # Loop over time steps.
    un = u.copy()
    vn = v.copy()

    # Step 4: Update u and v at interior points.
    u[1:, 1:] = un[1:, 1:] - un[1:, 1:]*dt/dx*(un[1:, 1:] - un[1:, :-1]) - vn[1:, 1:]*dt/dy*(un[1:, 1:] - un[:-1, 1:])
    v[1:, 1:] = vn[1:, 1:] - un[1:, 1:]*dt/dx*(vn[1:, 1:] - vn[1:, :-1]) - vn[1:, 1:]*dt/dy*(vn[1:, 1:] - vn[:-1, 1:])

    # Step 5: Apply the boundary conditions.
    u[:, 0] = u[:, -1] = 1
    u[0, :] = u[-1, :] = 1
    v[:, 0] = v[:, -1] = 1
    v[0, :] = v[-1, :] = 1

# Step 6: Plot the solution.
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots()
ax.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])
plt.show()