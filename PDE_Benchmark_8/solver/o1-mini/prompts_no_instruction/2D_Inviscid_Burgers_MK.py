import numpy as np

# Domain parameters
nx, ny = 151, 151
x_start, x_end = 0, 2
y_start, y_end = 0, 2
dx = (x_end - x_start) / (nx - 1)
dy = (y_end - y_start) / (ny - 1)
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
X, Y = np.meshgrid(x, y)

# Time parameters
nt = 300
sigma = 0.2
dt = sigma * min(dx, dy) / 2

# Initial conditions
u = np.ones((ny, nx))
v = np.ones((ny, nx))
u[np.logical_and.reduce((X >= 0.5, X <= 1, Y >= 0.5, Y <= 1))] = 2
v[np.logical_and.reduce((X >= 0.5, X <= 1, Y >= 0.5, Y <= 1))] = 2

# Apply boundary conditions
u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 1
v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 1

# Time-stepping loop
for _ in range(nt):
    # Predictor step
    u_star = np.copy(u)
    v_star = np.copy(v)
    
    u_star[1:-1,1:-1] = u[1:-1,1:-1] - dt * (
        u[1:-1,1:-1] * (u[1:-1,2:] - u[1:-1,1:-1]) / dx +
        v[1:-1,1:-1] * (u[2:,1:-1] - u[1:-1,1:-1]) / dy
    )
    
    v_star[1:-1,1:-1] = v[1:-1,1:-1] - dt * (
        u[1:-1,1:-1] * (v[1:-1,2:] - v[1:-1,1:-1]) / dx +
        v[1:-1,1:-1] * (v[2:,1:-1] - v[1:-1,1:-1]) / dy
    )
    
    # Apply boundary conditions to predictor
    u_star[0, :] = u_star[-1, :] = u_star[:, 0] = u_star[:, -1] = 1
    v_star[0, :] = v_star[-1, :] = v_star[:, 0] = v_star[:, -1] = 1
    
    # Corrector step
    u_new = np.copy(u)
    v_new = np.copy(v)
    
    u_new[1:-1,1:-1] = 0.5 * (
        u[1:-1,1:-1] + u_star[1:-1,1:-1] -
        dt * (
            u_star[1:-1,1:-1] * (u_star[1:-1,1:-1] - u_star[1:-1,0:-2]) / dx +
            v_star[1:-1,1:-1] * (u_star[1:-1,1:-1] - u_star[0:-2,1:-1]) / dy
        )
    )
    
    v_new[1:-1,1:-1] = 0.5 * (
        v[1:-1,1:-1] + v_star[1:-1,1:-1] -
        dt * (
            u_star[1:-1,1:-1] * (v_star[1:-1,1:-1] - v_star[1:-1,0:-2]) / dx +
            v_star[1:-1,1:-1] * (v_star[1:-1,1:-1] - v_star[0:-2,1:-1]) / dy
        )
    )
    
    # Update u and v
    u = u_new
    v = v_new
    
    # Apply boundary conditions to new u and v
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 1
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 1

# Save the final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/u_2D_Inviscid_Burgers_MK.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/v_2D_Inviscid_Burgers_MK.npy', v)