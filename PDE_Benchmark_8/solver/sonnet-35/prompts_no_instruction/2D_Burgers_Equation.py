import numpy as np

# Problem parameters
nx, ny = 41, 41
nt = 120
nu = 0.01
sigma = 0.0009
dx = 2.0 / (nx - 1)
dy = 2.0 / (ny - 1)
dt = sigma * dx * dy / nu

# Initialize arrays
u = np.ones((ny, nx)) 
v = np.ones((ny, nx))

# Initial condition modification
u[int(0.5/dy):int(1/dy)+1, int(0.5/dx):int(1/dx)+1] = 2
v[int(0.5/dy):int(1/dy)+1, int(0.5/dx):int(1/dx)+1] = 2

# Boundary conditions
def apply_boundary_conditions(u, v):
    u[:, 0] = 1
    u[:, -1] = 1
    u[0, :] = 1
    u[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1
    return u, v

# Finite difference solution
for _ in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # u momentum equation 
            u[i,j] = (un[i,j] - 
                      un[i,j] * dt/dx * (un[i,j] - un[i,j-1]) - 
                      vn[i,j] * dt/dy * (un[i,j] - un[i-1,j]) + 
                      nu * dt/dx**2 * (un[i,j+1] - 2*un[i,j] + un[i,j-1]) + 
                      nu * dt/dy**2 * (un[i+1,j] - 2*un[i,j] + un[i-1,j]))
            
            # v momentum equation
            v[i,j] = (vn[i,j] - 
                      un[i,j] * dt/dx * (vn[i,j] - vn[i,j-1]) - 
                      vn[i,j] * dt/dy * (vn[i,j] - vn[i-1,j]) + 
                      nu * dt/dx**2 * (vn[i,j+1] - 2*vn[i,j] + vn[i,j-1]) + 
                      nu * dt/dy**2 * (vn[i+1,j] - 2*vn[i,j] + vn[i-1,j]))
    
    # Apply boundary conditions
    u, v = apply_boundary_conditions(u, v)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_2D_Burgers_Equation.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/v_2D_Burgers_Equation.npy', v)