import numpy as np

# Parameters
alpha = 1.0  # thermal diffusivity
Q0 = 200.0  # source term coefficient (°C/s)
sigma = 0.1  # width of the source term
nx, ny = 41, 41  # number of grid points
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0
dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)
r = 0.2  # stability parameter
dt = r * dx**2 / alpha
t_max = 3.0  # maximum time
nt = int(t_max / dt)

# Grid
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initial condition
T = np.zeros((nx, ny))

# Time-stepping
for n in range(nt):
    Tn = T.copy()
    # Compute source term
    q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    # Update interior points
    T[1:-1,1:-1] = Tn[1:-1,1:-1] + dt * (
        alpha * (
            (Tn[2:,1:-1] - 2*Tn[1:-1,1:-1] + Tn[0:-2,1:-1]) / dx**2 +
            (Tn[1:-1,2:] - 2*Tn[1:-1,1:-1] + Tn[1:-1,0:-2]) / dy**2
        ) + q[1:-1,1:-1]
    )
    # Apply Dirichlet boundary conditions
    T[0,:] = 0.0
    T[-1,:] = 0.0
    T[:,0] = 0.0
    T[:,-1] = 0.0

# Save the final temperature field
np.save('T.npy', T)