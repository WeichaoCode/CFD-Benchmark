import numpy as np
import matplotlib.pyplot as plt

# Constants 
alpha = 0.01
Q0 = 200
sigma = 0.1
L = 1
W = 1
T_boundary = 0.
t_final = 0.5
CFL = 0.1

# Grid
N = 21
x = np.linspace(-L, L, N)
y = np.linspace(-W, W, N)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing='ij')

# Time parameters
dt = CFL * min(dx, dy)**2 / (4 * alpha)
Nt = int(t_final / dt)

# Source term
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Initialize solution: Temperature array
T = np.zeros((N, N), dtype=float)
T_new = np.zeros((N, N), dtype=float)

# Set boundary conditions
T[:,:] = T_boundary
T_new[:,:] = T_boundary

# Main time-stepping loop
for t in range(1, Nt+1):
    # Calculate new temperatures
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + dt*(
        alpha * ((T[:-2, 1:-1] - 2*T[1:-1, 1:-1] + T[2:, 1:-1])/dx**2 + 
                  (T[1:-1, :-2] - 2*T[1:-1, 1:-1] + T[1:-1, 2:])/dy**2) + q[1:-1, 1:-1])

    # Copy new temperature field to old one
    T[:, :] = T_new[:, :]

# Plotting
plt.figure(figsize=(8,6))
plt.contourf(X,Y,T,100,cmap='jet')
plt.title('2D Unsteady Heat Conduction', fontsize=15)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.colorbar(label='Temperature')
plt.show()