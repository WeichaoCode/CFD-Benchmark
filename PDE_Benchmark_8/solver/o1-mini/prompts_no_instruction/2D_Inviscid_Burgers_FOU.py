import numpy as np

# Parameters
nx = ny = 151
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
nt = 300
sigma = 0.2
dt = sigma * min(dx, dy) / 2

# Initialize velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial conditions: u = v = 2 in 0.5 <= x <=1 and 0.5 <= y <=1
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)
u[(X >= 0.5) & (X <=1) & (Y >=0.5) & (Y <=1)] = 2
v[(X >= 0.5) & (X <=1) & (Y >=0.5) & (Y <=1)] = 2

# Time-stepping
for n in range(nt):
    u_old = u.copy()
    v_old = v.copy()
    
    # Update u
    u[1:-1,1:-1] = (u_old[1:-1,1:-1] -
                    u_old[1:-1,1:-1] * dt / dx * (u_old[1:-1,1:-1] - u_old[1:-1,0:-2]) -
                    v_old[1:-1,1:-1] * dt / dy * (u_old[1:-1,1:-1] - u_old[0:-2,1:-1]))
    
    # Update v
    v[1:-1,1:-1] = (v_old[1:-1,1:-1] -
                    u_old[1:-1,1:-1] * dt / dx * (v_old[1:-1,1:-1] - v_old[1:-1,0:-2]) -
                    v_old[1:-1,1:-1] * dt / dy * (v_old[1:-1,1:-1] - v_old[0:-2,1:-1]))
    
    # Apply boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/u_2D_Inviscid_Burgers_FOU.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/v_2D_Inviscid_Burgers_FOU.npy', v)