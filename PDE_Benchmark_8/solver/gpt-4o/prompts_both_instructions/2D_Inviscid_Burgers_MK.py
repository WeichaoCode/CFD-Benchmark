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

# Initialize velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial conditions
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# MacCormack method
for n in range(nt):
    # Predictor step
    u_star = u.copy()
    v_star = v.copy()
    
    u_star[1:-1, 1:-1] = (u[1:-1, 1:-1] - dt * (
        u[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[1:-1, :-2]) / dx +
        v[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dy))
    
    v_star[1:-1, 1:-1] = (v[1:-1, 1:-1] - dt * (
        u[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[1:-1, :-2]) / dx +
        v[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dy))
    
    # Corrector step
    u[1:-1, 1:-1] = 0.5 * (u[1:-1, 1:-1] + u_star[1:-1, 1:-1] - dt * (
        u_star[1:-1, 1:-1] * (u_star[2:, 1:-1] - u_star[1:-1, 1:-1]) / dx +
        v_star[1:-1, 1:-1] * (u_star[1:-1, 2:] - u_star[1:-1, 1:-1]) / dy))
    
    v[1:-1, 1:-1] = 0.5 * (v[1:-1, 1:-1] + v_star[1:-1, 1:-1] - dt * (
        u_star[1:-1, 1:-1] * (v_star[2:, 1:-1] - v_star[1:-1, 1:-1]) / dx +
        v_star[1:-1, 1:-1] * (v_star[1:-1, 2:] - v_star[1:-1, 1:-1]) / dy))
    
    # Apply boundary conditions
    u[:, 0] = 1
    u[:, -1] = 1
    u[0, :] = 1
    u[-1, :] = 1
    
    v[:, 0] = 1
    v[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_2D_Inviscid_Burgers_MK.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/v_2D_Inviscid_Burgers_MK.npy', v)

# Visualization (optional)
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.quiver(X, Y, u, v)
plt.title('Velocity field at final time step')
plt.xlabel('x')
plt.ylabel('y')
plt.show()