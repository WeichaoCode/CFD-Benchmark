import numpy as np
import matplotlib.pyplot as plt

# Set the physical properties
mu = 1.0e-3
dPdz = -3.2

# Set the grid resolution
nx, ny = 51, 51
h = 0.1
dx = h / (nx - 1)
dy = h / (ny - 1)

# Create the grid
x = np.linspace(0, h, nx)
y = np.linspace(0, h, ny)
X, Y = np.meshgrid(x, y)

# Initialize the velocity field w
w = np.zeros((ny, nx))

# Set the coefficient values
aE = aW = mu * dy / dx
aN = aS = mu * dx / dy
aP = aE + aW + aN + aS
Su = dPdz * dx * dy

# Perform the Jacobi iteration
iters = 100
for it in range(iters):
    w_old = w.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            w[j, i] = (aE * w_old[j, i+1] + aW * w_old[j, i-1] +
                       aN * w_old[j-1, i] + aS * w_old[j+1, i] - Su) / aP

# Save the velocity field as npy file
np.save("velocity_field.npy", w)

# Plot the velocity field
plt.figure(figsize=(8, 8))
plt.contourf(X, Y, w, levels=100)
plt.colorbar(label='$w$')
plt.title('Contour plot of velocity field')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()