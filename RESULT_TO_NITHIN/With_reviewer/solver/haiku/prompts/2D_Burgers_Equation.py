import numpy as np

# Parameters
nx = ny = 50  # Number of grid points
Lx = Ly = 2.0  # Domain size
dx = Lx/(nx-1)
dy = Ly/(ny-1)
nu = 0.01  # Kinematic viscosity
dt = 0.0001  # Time step
t_final = 0.027

# Grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set initial conditions
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2.0
v[mask] = 2.0

# Arrays for next time step
u_new = np.zeros((ny, nx))
v_new = np.zeros((ny, nx))

t = 0
while t < t_final:
    # Set boundary conditions
    u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = 1.0
    v[0,:] = v[-1,:] = v[:,0] = v[:,-1] = 1.0
    
    # Interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # Central differences for spatial derivatives
            du_dx = (u[i,j+1] - u[i,j-1])/(2*dx)
            du_dy = (u[i+1,j] - u[i-1,j])/(2*dy)
            d2u_dx2 = (u[i,j+1] - 2*u[i,j] + u[i,j-1])/dx**2
            d2u_dy2 = (u[i+1,j] - 2*u[i,j] + u[i-1,j])/dy**2
            
            dv_dx = (v[i,j+1] - v[i,j-1])/(2*dx)
            dv_dy = (v[i+1,j] - v[i-1,j])/(2*dy)
            d2v_dx2 = (v[i,j+1] - 2*v[i,j] + v[i,j-1])/dx**2
            d2v_dy2 = (v[i+1,j] - 2*v[i,j] + v[i-1,j])/dy**2
            
            # Update velocities
            u_new[i,j] = u[i,j] + dt*(
                -u[i,j]*du_dx - v[i,j]*du_dy + 
                nu*(d2u_dx2 + d2u_dy2))
            
            v_new[i,j] = v[i,j] + dt*(
                -u[i,j]*dv_dx - v[i,j]*dv_dy + 
                nu*(d2v_dx2 + d2v_dy2))
    
    # Update solution
    u = u_new.copy()
    v = v_new.copy()
    t += dt

# Save final solutions
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Burgers_Equation.npy', u)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/v_2D_Burgers_Equation.npy', v)