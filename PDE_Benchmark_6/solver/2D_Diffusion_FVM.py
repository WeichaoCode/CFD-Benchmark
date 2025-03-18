import numpy as np
import matplotlib.pyplot as plt

# Constants
mu = 1e-3
rho = 1
dpdz = -3.2
h = 0.1
nx = ny = 51

# Grid spacing
dx = dy = h / (nx - 1)

# Initialize the velocity field
w = np.zeros((nx, ny))

# Coefficients
aE = aW = mu * dy / dx
aN = aS = mu * dx / dy
aP = aE + aW + aN + aS
Su = dpdz * dx * dy

# Jacobi iteration
for _ in range(5000):
    w_old = w.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            w[i, j] = (aE * w_old[i+1, j] + aW * w_old[i-1, j] + aN * w_old[i, j+1] + aS * w_old[i, j-1] - Su) / aP

# Visualization
plt.figure(figsize=(8, 6))
plt.contourf(w, levels=50, cmap='jet')
plt.colorbar(label='Velocity (m/s)')
plt.title('Velocity distribution in a square duct')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Save the velocity field
np.save('velocity_field.npy', w)