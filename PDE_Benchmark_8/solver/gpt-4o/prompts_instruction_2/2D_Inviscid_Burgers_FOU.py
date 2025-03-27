import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 151, 151
nt = 300
sigma = 0.2
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = sigma * min(dx, dy) / 2

# Initialize u and v
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial conditions
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update u and v using First-Order Upwind scheme
    u[1:, 1:] = (un[1:, 1:] - 
                 dt / dx * un[1:, 1:] * (un[1:, 1:] - un[1:, :-1]) - 
                 dt / dy * vn[1:, 1:] * (un[1:, 1:] - un[:-1, 1:]))
    
    v[1:, 1:] = (vn[1:, 1:] - 
                 dt / dx * un[1:, 1:] * (vn[1:, 1:] - vn[1:, :-1]) - 
                 dt / dy * vn[1:, 1:] * (vn[1:, 1:] - vn[:-1, 1:]))
    
    # Apply boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/u_2D_Inviscid_Burgers_FOU.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/v_2D_Inviscid_Burgers_FOU.npy', v)

# Visualization
plt.figure(figsize=(8, 6))
plt.quiver(x, y, u, v)
plt.title('Velocity field at final time step')
plt.xlabel('x')
plt.ylabel('y')
plt.show()