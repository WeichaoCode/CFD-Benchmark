import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 151, 151
nt = 300
sigma = 0.2
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = sigma * min(dx, dy) / 2

# Grid
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Initial Conditions
u = np.ones((ny, nx))
v = np.ones((ny, nx))
mask = (X >= 0.5) & (X <=1) & (Y >=0.5) & (Y <=1)
u[mask] = 2
v[mask] = 2

# Time-stepping
for _ in range(nt):
    # Predictor step for u
    u_star = u.copy()
    du_dx = (u[1:-1,1:-1] - u[1:-1,0:-2]) / dx
    du_dy = (u[1:-1,1:-1] - u[0:-2,1:-1]) / dy
    u_star[1:-1,1:-1] = u[1:-1,1:-1] - dt * (u[1:-1,1:-1]*du_dx + v[1:-1,1:-1]*du_dy)
    
    # Predictor step for v
    v_star = v.copy()
    dv_dx = (v[1:-1,1:-1] - v[1:-1,0:-2]) / dx
    dv_dy = (v[1:-1,1:-1] - v[0:-2,1:-1]) / dy
    v_star[1:-1,1:-1] = v[1:-1,1:-1] - dt * (u[1:-1,1:-1]*dv_dx + v[1:-1,1:-1]*dv_dy)
    
    # Corrector step for u
    du_star_dx = (u_star[1:-1,2:] - u_star[1:-1,1:-1]) / dx
    du_star_dy = (u_star[2:,1:-1] - u_star[1:-1,1:-1]) / dy
    u_new = 0.5 * (u[1:-1,1:-1] + u_star[1:-1,1:-1] - dt * (u_star[1:-1,1:-1]*du_star_dx + v_star[1:-1,1:-1]*du_star_dy))
    
    # Corrector step for v
    dv_star_dx = (v_star[1:-1,2:] - v_star[1:-1,1:-1]) / dx
    dv_star_dy = (v_star[2:,1:-1] - v_star[1:-1,1:-1]) / dy
    v_new = 0.5 * (v[1:-1,1:-1] + v_star[1:-1,1:-1] - dt * (u_star[1:-1,1:-1]*dv_star_dx + v_star[1:-1,1:-1]*dv_star_dy))
    
    # Update u and v
    u[1:-1,1:-1] = u_new
    v[1:-1,1:-1] = v_new
    
    # Apply boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Visualization
plt.figure(figsize=(8,6))
plt.quiver(X, Y, u, v)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Velocity Field at Final Time Step')
plt.show()

# Save final solutions
np.save('u.npy', u)
np.save('v.npy', v)