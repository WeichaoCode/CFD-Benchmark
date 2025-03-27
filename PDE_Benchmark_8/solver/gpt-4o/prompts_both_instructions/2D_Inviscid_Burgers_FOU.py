import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 151, 151
nt = 300
Lx, Ly = 2.0, 2.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
sigma = 0.2
dt = sigma * min(dx, dy) / 2

# Initialize the velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial conditions: hat function
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update u and v using the First-Order Upwind scheme
    u[1:, 1:] = (un[1:, 1:] - dt * (un[1:, 1:] * (un[1:, 1:] - un[1:, :-1]) / dx +
                                    vn[1:, 1:] * (un[1:, 1:] - un[:-1, 1:]) / dy))
    v[1:, 1:] = (vn[1:, 1:] - dt * (un[1:, 1:] * (vn[1:, 1:] - vn[1:, :-1]) / dx +
                                    vn[1:, 1:] * (vn[1:, 1:] - vn[:-1, 1:]) / dy))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_2D_Inviscid_Burgers_FOU.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/v_2D_Inviscid_Burgers_FOU.npy', v)

# Visualization (optional)
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, u, v)
plt.title('Velocity field at final time step')
plt.xlabel('x')
plt.ylabel('y')
plt.show()