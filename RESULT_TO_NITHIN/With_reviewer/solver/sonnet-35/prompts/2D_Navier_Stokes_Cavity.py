import numpy as np

# Problem parameters
Lx, Ly = 2.0, 2.0
rho = 1.0
nu = 0.1
t_end = 10.0

# Discretization parameters
nx, ny = 100, 100
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = 0.001  # Smaller time step

# Grid generation
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Boundary conditions
u[-1, :] = 1.0  # Top lid moving

def central_difference_x(field):
    dx_field = np.zeros_like(field)
    dx_field[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2*dx)
    # Forward/backward differences at boundaries
    dx_field[:, 0] = (field[:, 1] - field[:, 0]) / dx
    dx_field[:, -1] = (field[:, -1] - field[:, -2]) / dx
    return dx_field

def central_difference_y(field):
    dy_field = np.zeros_like(field)
    dy_field[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2*dy)
    # Forward/backward differences at boundaries
    dy_field[0, :] = (field[1, :] - field[0, :]) / dy
    dy_field[-1, :] = (field[-1, :] - field[-2, :]) / dy
    return dy_field

def laplacian(field):
    lap = np.zeros_like(field)
    lap[1:-1, 1:-1] = (
        (field[1:-1, 2:] + field[1:-1, :-2] - 2*field[1:-1, 1:-1]) / dx**2 +
        (field[2:, 1:-1] + field[:-2, 1:-1] - 2*field[1:-1, 1:-1]) / dy**2
    )
    return lap

# Time integration
for t in np.arange(0, t_end, dt):
    # Compute derivatives
    du_dx = central_difference_x(u)
    du_dy = central_difference_y(u)
    dv_dx = central_difference_x(v)
    dv_dy = central_difference_y(v)
    
    # Momentum equations with stabilization
    u_adv = -u * du_dx - v * du_dy
    v_adv = -u * dv_dx - v * dv_dy
    
    u_diff = nu * laplacian(u)
    v_diff = nu * laplacian(v)
    
    u_new = u + dt * (u_adv + u_diff)
    v_new = v + dt * (v_adv + v_diff)
    
    # Enforce boundary conditions
    u_new[0, :] = 0  # Bottom wall
    u_new[:, 0] = 0  # Left wall
    u_new[:, -1] = 0  # Right wall
    u_new[-1, :] = 1.0  # Top lid
    
    v_new[0, :] = 0  # Bottom wall
    v_new[:, 0] = 0  # Left wall
    v_new[:, -1] = 0  # Right wall
    v_new[-1, :] = 0  # Top lid
    
    # Pressure Poisson equation
    rhs = -rho * (du_dx**2 + 2*du_dy*dv_dx + dv_dy**2)
    
    # Solve pressure Poisson equation using Jacobi iteration
    p_new = p.copy()
    for _ in range(50):  # Fewer iterations, more stable
        p_old = p_new.copy()
        p_new[1:-1, 1:-1] = 0.25 * (
            p_new[1:-1, 2:] + p_new[1:-1, :-2] + 
            p_new[2:, 1:-1] + p_new[:-2, 1:-1] - 
            dx**2 * rhs[1:-1, 1:-1]
        )
        
        # Pressure boundary conditions
        p_new[0, :] = p_new[1, :]  # Neumann at bottom
        p_new[-1, :] = 0  # Dirichlet at top
        p_new[:, 0] = p_new[:, 1]  # Neumann at left
        p_new[:, -1] = p_new[:, -2]  # Neumann at right
        
        # Convergence check with relative tolerance
        if np.max(np.abs(p_new - p_old)) / (np.max(np.abs(p_new)) + 1e-10) < 1e-4:
            break
    
    # Update fields
    u = u_new
    v = v_new
    p = p_new

# Save final solutions
np.save('/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_2D_Navier_Stokes_Cavity.npy', u)
np.save('/PDE_Benchmark/results/prediction/sonnet-35/prompts/v_2D_Navier_Stokes_Cavity.npy', v)
np.save('/PDE_Benchmark/results/prediction/sonnet-35/prompts/p_2D_Navier_Stokes_Cavity.npy', p)