import numpy as np

# Parameters
alpha = 1.0
Q0 = 200.0
sigma = 0.1
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_final = 3.0

# Grid
N = 101
x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, N)
dx = (x_max - x_min) / (N - 1)
dy = (y_max - y_min) / (N - 1)
X, Y = np.meshgrid(x, y)

# Initial condition
T = 1.0 + 200.0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Boundary conditions
T[0, :] = 1.0
T[-1, :] = 1.0
T[:, 0] = 1.0
T[:, -1] = 1.0

# Source term (time-independent)
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Time step based on stability condition for explicit scheme
dt = min(dx, dy)**2 / (4 * alpha)
nt = int(t_final / dt) + 1
dt = t_final / nt  # Adjust dt to reach t_final exactly

# Time integration
for _ in range(nt):
    T_new = T.copy()
    T_new[1:-1,1:-1] = T[1:-1,1:-1] + dt * (
        alpha * (
            (T[2:,1:-1] - 2*T[1:-1,1:-1] + T[0:-2,1:-1]) / dx**2 +
            (T[1:-1,2:] - 2*T[1:-1,1:-1] + T[1:-1,0:-2]) / dy**2
        ) + q[1:-1,1:-1]
    )
    # Apply boundary conditions
    T_new[0, :] = 1.0
    T_new[-1, :] = 1.0
    T_new[:, 0] = 1.0
    T_new[:, -1] = 1.0
    T = T_new

# Save the final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/T_2D_Unsteady_Heat_Equation.npy', T)