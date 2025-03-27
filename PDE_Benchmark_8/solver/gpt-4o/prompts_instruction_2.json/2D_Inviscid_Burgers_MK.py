import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = ny = 151
nt = 300
sigma = 0.2
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = sigma * min(dx, dy) / 2

# Initialize velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial conditions
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# MacCormack method
for n in range(nt):
    # Predictor step
    u_pred = u.copy()
    v_pred = v.copy()
    
    u_pred[1:-1, 1:-1] = (u[1:-1, 1:-1] - 
                          dt / dx * u[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[1:-1, :-2]) - 
                          dt / dy * v[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[:-2, 1:-1]))
    
    v_pred[1:-1, 1:-1] = (v[1:-1, 1:-1] - 
                          dt / dx * u[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[1:-1, :-2]) - 
                          dt / dy * v[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[:-2, 1:-1]))
    
    # Corrector step
    u[1:-1, 1:-1] = (0.5 * (u[1:-1, 1:-1] + u_pred[1:-1, 1:-1] - 
                            dt / dx * u_pred[1:-1, 1:-1] * (u_pred[1:-1, 2:] - u_pred[1:-1, 1:-1]) - 
                            dt / dy * v_pred[1:-1, 1:-1] * (u_pred[2:, 1:-1] - u_pred[1:-1, 1:-1])))
    
    v[1:-1, 1:-1] = (0.5 * (v[1:-1, 1:-1] + v_pred[1:-1, 1:-1] - 
                            dt / dx * u_pred[1:-1, 1:-1] * (v_pred[1:-1, 2:] - v_pred[1:-1, 1:-1]) - 
                            dt / dy * v_pred[1:-1, 1:-1] * (v_pred[2:, 1:-1] - v_pred[1:-1, 1:-1])))
    
    # Enforce boundary conditions
    u[:, 0] = 1
    u[:, -1] = 1
    u[0, :] = 1
    u[-1, :] = 1
    
    v[:, 0] = 1
    v[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1

# Save the final solution
np.save('u_final.npy', u)
np.save('v_final.npy', v)

# Visualization
plt.figure(figsize=(8, 6))
plt.quiver(x, y, u, v)
plt.title('Velocity Field at Final Time Step')
plt.xlabel('x')
plt.ylabel('y')
plt.show()