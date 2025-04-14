import numpy as np

# Problem parameters
Lx, Ly = 1.0, 1.0  # Domain dimensions
rho = 1.0  # Density
nu = 0.1  # Kinematic viscosity
dt = 0.01  # Increased time step
T = 5  # Further reduced simulation time
nx, ny = 30, 30  # Reduced grid resolution

# Grid generation
dx, dy = Lx/(nx-1), Ly/(ny-1)

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Boundary conditions
u[-1, :] = 1.0  # Top lid moving with unit velocity

# Simplified semi-implicit time-stepping
def compute_velocity(u_old, v_old, is_u_velocity):
    field = np.zeros_like(u_old)
    
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if is_u_velocity:
                # U-velocity update
                conv_x = (
                    (u_old[i, j] + u_old[i, j+1])**2/2 - 
                    (u_old[i, j-1] + u_old[i, j])**2/2
                ) / dx
                
                conv_y = (
                    (u_old[i, j] + u_old[i+1, j]) * 
                    (v_old[i, j] + v_old[i, j+1])/2 - 
                    (u_old[i-1, j] + u_old[i, j]) * 
                    (v_old[i-1, j] + v_old[i-1, j+1])/2
                ) / dy
                
                diff = nu * (
                    (u_old[i, j+1] - 2*u_old[i, j] + u_old[i, j-1]) / dx**2 +
                    (u_old[i+1, j] - 2*u_old[i, j] + u_old[i-1, j]) / dy**2
                )
            else:
                # V-velocity update
                conv_x = (
                    (u_old[i, j] + u_old[i, j+1]) * 
                    (v_old[i, j] + v_old[i, j+1])/2 - 
                    (u_old[i, j-1] + u_old[i, j]) * 
                    (v_old[i, j-1] + v_old[i, j])/2
                ) / dx
                
                conv_y = (
                    (v_old[i, j] + v_old[i+1, j])**2/2 - 
                    (v_old[i-1, j] + v_old[i, j])**2/2
                ) / dy
                
                diff = nu * (
                    (v_old[i, j+1] - 2*v_old[i, j] + v_old[i, j-1]) / dx**2 +
                    (v_old[i+1, j] - 2*v_old[i, j] + v_old[i-1, j]) / dy**2
                )
            
            field[i, j] = u_old[i, j] - dt * (conv_x + conv_y) + dt * diff
    
    return field

# Time-stepping
for _ in range(int(T/dt)):
    # Store old velocities
    u_old = u.copy()
    v_old = v.copy()
    
    # Update velocities
    u = compute_velocity(u_old, v_old, is_u_velocity=True)
    v = compute_velocity(v_old, u_old, is_u_velocity=False)
    
    # Enforce boundary conditions
    u[-1, :] = 1.0  # Top lid
    u[0, :] = 0.0   # Bottom wall
    u[:, 0] = 0.0   # Left wall
    u[:, -1] = 0.0  # Right wall
    
    v[-1, :] = 0.0  # Top lid
    v[0, :] = 0.0   # Bottom wall
    v[:, 0] = 0.0   # Left wall
    v[:, -1] = 0.0  # Right wall

# Save final solutions
save_values = ['u', 'v', 'p']
for var in save_values:
    np.save(f'{var}_final.npy', locals()[var])