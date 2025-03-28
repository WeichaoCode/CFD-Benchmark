import numpy as np

# Parameters
nx, ny = 151, 151
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
sigma = 0.2
dt = sigma * min(dx, dy) / 2
nt = 300

# Initialize u and v
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial conditions
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping using MacCormack method
for n in range(nt):
    # Predictor step
    u_star = u.copy()
    v_star = v.copy()
    
    u_star[1:-1, 1:-1] = (u[1:-1, 1:-1] - 
                          dt / dx * u[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[1:-1, :-2]) - 
                          dt / dy * v[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[:-2, 1:-1]))
    
    v_star[1:-1, 1:-1] = (v[1:-1, 1:-1] - 
                          dt / dx * u[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[1:-1, :-2]) - 
                          dt / dy * v[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[:-2, 1:-1]))
    
    # Corrector step
    u[1:-1, 1:-1] = (0.5 * (u[1:-1, 1:-1] + u_star[1:-1, 1:-1] - 
                            dt / dx * u_star[1:-1, 1:-1] * (u_star[1:-1, 2:] - u_star[1:-1, 1:-1]) - 
                            dt / dy * v_star[1:-1, 1:-1] * (u_star[2:, 1:-1] - u_star[1:-1, 1:-1])))
    
    v[1:-1, 1:-1] = (0.5 * (v[1:-1, 1:-1] + v_star[1:-1, 1:-1] - 
                            dt / dx * u_star[1:-1, 1:-1] * (v_star[1:-1, 2:] - v_star[1:-1, 1:-1]) - 
                            dt / dy * v_star[1:-1, 1:-1] * (v_star[2:, 1:-1] - v_star[1:-1, 1:-1])))
    
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
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_2D_Inviscid_Burgers_MK.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/v_2D_Inviscid_Burgers_MK.npy', v)