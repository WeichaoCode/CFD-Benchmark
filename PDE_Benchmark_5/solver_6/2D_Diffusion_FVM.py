import numpy as np
import matplotlib.pyplot as plt

h = 0.1
nx, ny = 50, 50
dP_dz = -3.2
mu = 1e-3
tol = 1e-6
max_it = 500

dx = h/nx
dy = h/ny

w_old = np.zeros((nx+1, ny+1))
w_new = np.zeros((nx+1, ny+1))

# Discretization based on central difference and Finite Volume Method
for it in range(max_it):
    for i in range(1, nx):
        for j in range(1, ny):
            a_E = mu/dx**2
            a_W = mu/dx**2
            a_N = mu/dy**2
            a_S = mu/dy**2

            a_P = a_E + a_W + a_N + a_S + (dP_dz/mu)

            w_new[i,j] = (a_E*w_old[i+1,j] + a_W*w_old[i-1,j] + a_N*w_old[i,j+1] + a_S*w_old[i,j-1] + dP_dz)/a_P

    if np.abs(w_new - w_old).max() < tol:
        break

    w_old = w_new.copy()

# Plotting
plt.contourf(w_new, cmap = "coolwarm")
plt.title("Contour of w-velocity")
plt.colorbar(label="w-velocity")
plt.show()