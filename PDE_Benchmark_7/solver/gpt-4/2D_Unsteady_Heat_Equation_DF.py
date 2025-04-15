import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
nx, ny = 41, 41
t_max = 3.0
alpha = 1.0
sigma = 0.1
Q0 = 200.0
dx = 2.0 / (nx - 1)
dy = 2.0 / (ny - 1)
beta = dx / dy
r = 0.25
r = r / (1 + beta**2)
dt = r * dx**2 / alpha

# Initialize the solution arrays
T = np.zeros((nx, ny, 3))  # current solution
T_new = np.zeros((nx, ny))  # new solution

# Define the source term function
def q(x, y):
    return Q0 * np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Time loop
t = 0.0
while t < t_max:
    # Compute new solution
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            T_new[i, j] = (2*r*(T[i+1, j, 1] + T[i-1, j, 1]) + 2*beta**2*r*(T[i, j+1, 1] + T[i, j-1, 1]) + T[i, j, 0] + 2*dt*q(i*dx, j*dy)) / (1 + 2*r + 2*beta**2*r)
    # Update solution arrays
    T[:, :, 0] = T[:, :, 1]
    T[:, :, 1] = T[:, :, 2]
    T[:, :, 2] = T_new
    # Advance time
    t += dt

# Save the solution to a .npy file
np.save('solution.npy', T_new)

# Visualize the final temperature distribution
plt.figure(figsize=(8, 6))
plt.imshow(T_new, cmap='hot', extent=[-1, 1, -1, 1])
plt.colorbar(label='Temperature (°C)')
plt.title('Final Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()