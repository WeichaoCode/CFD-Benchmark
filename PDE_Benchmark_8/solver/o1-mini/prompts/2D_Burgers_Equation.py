import numpy as np

# Parameters
nu = 0.01
x_start, x_end = 0.0, 2.0
y_start, y_end = 0.0, 2.0
t_final = 0.027
nx, ny = 101, 101
dx = (x_end - x_start) / (nx - 1)
dy = (y_end - y_start) / (ny - 1)

# Stability condition for time step
sigma = 0.2
dt = sigma * min(dx, dy)**2 / nu
nt = int(t_final / dt) + 1
dt = t_final / nt

# Create grid
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
X, Y = np.meshgrid(x, y)

# Initialize velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Apply initial conditions: u = v = 2 for 0.5 <= x <=1 and 0.5 <= y <=1
u_initial_region = np.where((X >= 0.5) & (X <=1.0) & (Y >=0.5) & (Y <=1.0))
v_initial_region = np.where((X >= 0.5) & (X <=1.0) & (Y >=0.5) & (Y <=1.0))
u[u_initial_region] = 2.0
v[v_initial_region] = 2.0

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Compute derivatives
    # Interior points
    u_x = (un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx)
    u_y = (un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy)
    v_x = (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dx)
    v_y = (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)

    u_xx = (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) / dx**2
    u_yy = (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dy**2
    v_xx = (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) / dx**2
    v_yy = (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]) / dy**2

    # Update velocity fields
    u[1:-1,1:-1] = un[1:-1,1:-1] + dt * (
        - un[1:-1,1:-1] * u_x
        - vn[1:-1,1:-1] * u_y
        + nu * (u_xx + u_yy)
    )

    v[1:-1,1:-1] = vn[1:-1,1:-1] + dt * (
        - un[1:-1,1:-1] * v_x
        - vn[1:-1,1:-1] * v_y
        + nu * (v_xx + v_yy)
    )
    
    # Apply Dirichlet boundary conditions: u = v =1 on all boundaries
    u[0, :] = 1.0
    u[-1, :] = 1.0
    u[:, 0] = 1.0
    u[:, -1] = 1.0

    v[0, :] = 1.0
    v[-1, :] = 1.0
    v[:, 0] = 1.0
    v[:, -1] = 1.0

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/u_2D_Burgers_Equation.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/v_2D_Burgers_Equation.npy', v)