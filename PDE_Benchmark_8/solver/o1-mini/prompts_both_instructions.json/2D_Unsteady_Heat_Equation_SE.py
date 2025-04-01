import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0
Q0 = 200.0
sigma = 0.1
x_min, x_max, nx = -1.0, 1.0, 41
y_min, y_max, ny = -1.0, 1.0, 41
t_max = 3.0
beta = 1.0
r = 0.25  # Stability condition: (1 + beta^2) * r <= 0.5

# Grid setup
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
dt = r * dx**2 / alpha

# Initialize temperature field
X, Y = np.meshgrid(x, y)
T = 1.0 + Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Time stepping
n_steps = int(t_max / dt)
for _ in range(n_steps):
    T_new = T.copy()
    T_new[1:-1,1:-1] = (r * (T[1:-1,2:] - 2*T[1:-1,1:-1] + T[1:-1,0:-2]) +
                        beta**2 * r * (T[2:,1:-1] - 2*T[1:-1,1:-1] + T[0:-2,1:-1]) +
                        T[1:-1,1:-1] + dt * Q0 * np.exp(-(X[1:-1,1:-1]**2 + Y[1:-1,1:-1]**2) / (2 * sigma**2)))
    T_new[0,:] = 1.0
    T_new[-1,:] = 1.0
    T_new[:,0] = 1.0
    T_new[:,-1] = 1.0
    T = T_new

# Save the final temperature field
T_final = T
np.save('T.npy', T_final)

# Plot the final temperature field
plt.contourf(X, Y, T_final, 50, cmap='hot')
plt.colorbar(label='Temperature')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Temperature Distribution at t = {:.2f} s'.format(t_max))
plt.show()