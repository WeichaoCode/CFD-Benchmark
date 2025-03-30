import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0          # Thermal diffusivity
Q0 = 200.0           # Source term coefficient (°C/s)
sigma = 0.1          # Standard deviation for source term
nx, ny = 41, 41      # Number of grid points
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_max = 3.0          # Maximum time (s)

# Spatial discretization
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Stability condition
beta = dy / dx
r = 1.0 / (2.0 * (1.0 + beta**2))  # r <= 1/(2*(1 + beta^2))
dt = r * dx**2 / alpha

# Time steps
nt = int(t_max / dt)

# Initialize temperature field
T = np.zeros((ny, nx))

# Define source term (independent of time)
q = Q0 * np.exp(- (X**2 + Y**2) / (2.0 * sigma**2))

# Time-stepping loop
for n in range(nt):
    Tn = T.copy()
    # Update internal points
    T[1:-1,1:-1] = (r * (Tn[1:-1,2:] - 2.0 * Tn[1:-1,1:-1] + Tn[1:-1,0:-2]) +
                    beta**2 * r * (Tn[2:,1:-1] - 2.0 * Tn[1:-1,1:-1] + Tn[0:-2,1:-1]) +
                    Tn[1:-1,1:-1] + dt * q[1:-1,1:-1])
    # Apply Dirichlet boundary conditions
    T[0, :] = 0.0
    T[-1, :] = 0.0
    T[:, 0] = 0.0
    T[:, -1] = 0.0

# Save the final temperature field
np.save('T.npy', T)

# Plot the final temperature field
plt.figure(figsize=(8,6))
cp = plt.contourf(X, Y, T, 50, cmap='hot')
plt.colorbar(cp)
plt.title('Final Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()