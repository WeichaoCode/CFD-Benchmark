import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 41, 41
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
alpha = 0.01  # Reduced alpha to ensure stability
Q0 = 200.0
sigma = 0.1
beta = dy / dx  # Assuming dx = dy, so beta = 1.0
r = 0.001  # Further reduced r to ensure stability
dt = r * dx**2 / alpha
t_max = 3.0

# Create meshgrid
X, Y = np.meshgrid(x, y)

# Source term
q = Q0 * np.exp(- (X**2 + Y**2) / (2 * sigma**2))

# Initialize temperature fields
T_prev = np.zeros((ny, nx))
T_current = np.zeros((ny, nx))
T_new = np.zeros((ny, nx))

# Function to apply Dirichlet boundary conditions
def apply_boundary(T):
    T[0, :] = 0
    T[-1, :] = 0
    T[:, 0] = 0
    T[:, -1] = 0
    return T

# Apply boundary conditions to initial fields
T_prev = apply_boundary(T_prev)
T_current = apply_boundary(T_current)

# First time step using Forward Euler method
T_current[1:-1,1:-1] = T_prev[1:-1,1:-1] + dt * (
    alpha * (
        (T_prev[2:,1:-1] - 2*T_prev[1:-1,1:-1] + T_prev[:-2,1:-1]) / dx**2 +
        (T_prev[1:-1,2:] - 2*T_prev[1:-1,1:-1] + T_prev[1:-1,:-2]) / dy**2
    ) + q[1:-1,1:-1]
)

# Apply boundary conditions
T_current = apply_boundary(T_current)

# Time stepping using DuFort-Frankel method
t = dt
while t < t_max:
    T_new[1:-1,1:-1] = (
        2 * r * (
            T_current[2:,1:-1] + T_current[:-2,1:-1] +
            T_current[1:-1,2:] + T_current[1:-1,:-2]
        ) + 
        T_prev[1:-1,1:-1] + 
        2 * dt * q[1:-1,1:-1]
    ) / (1 + 4 * r)
    
    # Apply Dirichlet boundary conditions
    T_new = apply_boundary(T_new)
    
    # Update for next time step
    T_prev, T_current = T_current, T_new.copy()
    t += dt

# Final temperature field
T = T_current

# Save the final temperature field
np.save('T.npy', T)

# Visualization
plt.figure(figsize=(6,5))
contour = plt.contourf(X, Y, T, 50, cmap='hot')
plt.colorbar(contour)
plt.title(f'Temperature Distribution at t = {t_max:.2f} s')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()