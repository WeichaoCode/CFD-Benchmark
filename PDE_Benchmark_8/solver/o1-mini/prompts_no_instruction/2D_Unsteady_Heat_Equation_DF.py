import numpy as np

# Parameters
alpha = 1.0
Q0 = 200.0
sigma = 0.1
r = 0.4

# Domain
nx = 41
ny = 41
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Time step
dt = r * dx**2 / alpha
t_max = 3.0
nt = int(t_max / dt)

# Grid
X, Y = np.meshgrid(x, y, indexing='ij')

# Source term
q = Q0 * np.exp(- (X**2 + Y**2) / (2 * sigma**2))

# Initial condition
T_prev = 1.0 + Q0 * np.exp(- (X**2 + Y**2) / (2 * sigma**2))

# Apply boundary conditions
T_prev[0, :] = 1.0
T_prev[-1, :] = 1.0
T_prev[:, 0] = 1.0
T_prev[:, -1] = 1.0

# Initialize T_current using Forward Euler
T_current = np.copy(T_prev)
T_current[1:-1, 1:-1] = T_prev[1:-1, 1:-1] + dt * (
    alpha * (
        (T_prev[2:, 1:-1] - 2 * T_prev[1:-1, 1:-1] + T_prev[0:-2, 1:-1]) / dx**2 +
        (T_prev[1:-1, 2:] - 2 * T_prev[1:-1, 1:-1] + T_prev[1:-1, 0:-2]) / dy**2
    ) + q[1:-1, 1:-1]
)

# Apply boundary conditions
T_current[0, :] = 1.0
T_current[-1, :] = 1.0
T_current[:, 0] = 1.0
T_current[:, -1] = 1.0

# Time stepping using DuFort-Frankel
for _ in range(1, nt):
    T_new = np.zeros_like(T_prev)
    T_new[1:-1, 1:-1] = (
        (1 - 4 * r) * T_prev[1:-1, 1:-1] +
        2 * r * (T_current[2:, 1:-1] + T_current[0:-2, 1:-1] +
                 T_current[1:-1, 2:] + T_current[1:-1, 0:-2]) +
        2 * dt * q[1:-1, 1:-1]
    ) / (1 + 4 * r)
    
    # Apply boundary conditions
    T_new[0, :] = 1.0
    T_new[-1, :] = 1.0
    T_new[:, 0] = 1.0
    T_new[:, -1] = 1.0
    
    # Update for next step
    T_prev, T_current = T_current, T_new

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/T_current_2D_Unsteady_Heat_Equation_DF.npy', T_current)