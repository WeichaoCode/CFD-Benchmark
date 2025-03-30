import numpy as np

# Domain parameters
nx, ny = 151, 151
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Time parameters
nt = 300
sigma = 0.2
dt = sigma * min(dx, dy) / 2

# Initialize velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Apply initial conditions
u[np.where((X >= 0.5) & (X <= 1) & (Y >= 0.5) & (Y <= 1))] = 2
v[np.where((X >= 0.5) & (X <= 1) & (Y >= 0.5) & (Y <= 1))] = 2

# Function to apply boundary conditions
def apply_boundary_conditions(u, v):
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1
    return u, v

# Apply initial boundary conditions
u, v = apply_boundary_conditions(u, v)

# Time-stepping loop using MacCormack Method
for _ in range(nt):
    # Predictor step
    u_pred = np.copy(u)
    v_pred = np.copy(v)
    
    u_pred[1:-1,1:-1] = u[1:-1,1:-1] - dt * (
        u[1:-1,1:-1] * (u[1:-1,1:-1] - u[1:-1,0:-2]) / dx +
        v[1:-1,1:-1] * (u[1:-1,1:-1] - u[0:-2,1:-1]) / dy
    )
    
    v_pred[1:-1,1:-1] = v[1:-1,1:-1] - dt * (
        u[1:-1,1:-1] * (v[1:-1,1:-1] - v[1:-1,0:-2]) / dx +
        v[1:-1,1:-1] * (v[1:-1,1:-1] - v[0:-2,1:-1]) / dy
    )
    
    # Apply boundary conditions to predictor
    u_pred, v_pred = apply_boundary_conditions(u_pred, v_pred)
    
    # Corrector step
    u_corr = np.copy(u)
    v_corr = np.copy(v)
    
    u_corr[1:-1,1:-1] = 0.5 * (u[1:-1,1:-1] + u_pred[1:-1,1:-1]) - 0.5 * dt * (
        u_pred[1:-1,1:-1] * (u_pred[2:,1:-1] - u_pred[0:-2,1:-1]) / (2*dx) +
        v_pred[1:-1,1:-1] * (u_pred[1:-1,2:] - u_pred[1:-1,0:-2]) / (2*dy)
    )
    
    v_corr[1:-1,1:-1] = 0.5 * (v[1:-1,1:-1] + v_pred[1:-1,1:-1]) - 0.5 * dt * (
        u_pred[1:-1,1:-1] * (v_pred[2:,1:-1] - v_pred[0:-2,1:-1]) / (2*dx) +
        v_pred[1:-1,1:-1] * (v_pred[1:-1,2:] - v_pred[1:-1,0:-2]) / (2*dy)
    )
    
    # Update the solution
    u[1:-1,1:-1] = u_corr[1:-1,1:-1]
    v[1:-1,1:-1] = v_corr[1:-1,1:-1]
    
    # Apply boundary conditions
    u, v = apply_boundary_conditions(u, v)

# Save the final solution
np.save('u.npy', u)
np.save('v.npy', v)