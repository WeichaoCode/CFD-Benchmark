import numpy as np

# Parameters
Q0 = 200.0
sigma = 0.1
alpha = 1.0
nx, ny = 41, 41
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
r = 0.1
dt = r * dx**2 / alpha
t_max = 3.0
nt = int(t_max / dt)

# Initialize temperature field
X, Y = np.meshgrid(x, y, indexing='ij')
T = 1.0 + 200.0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Apply boundary conditions
T[0, :] = 1.0
T[-1, :] = 1.0
T[:, 0] = 1.0
T[:, -1] = 1.0

# Precompute constants
rx = alpha * dt / (2 * dx**2)
ry = alpha * dt / (2 * dy**2)

# Thomas algorithm for tridiagonal systems
def thomas(a, b, c, d):
    nf = len(d)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for it in range(1, nf):
        mc = ac[it-1] / bc[it-1]
        bc[it] = bc[it] - mc * cc[it-1]
        dc[it] = dc[it] - mc * dc[it-1]
    xc = bc
    xc[-1] = dc[-1] / bc[-1]
    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il+1]) / bc[il]
    return xc

# Time-stepping loop
for _ in range(nt):
    # First half-step (x-direction implicit)
    T_star = T.copy()
    for j in range(1, ny-1):
        a = -rx * np.ones(nx-2)
        b = (1 + 2*rx) * np.ones(nx-2)
        c = -rx * np.ones(nx-2)
        d = T[1:-1, j] + ry * (T[1:-1, j+1] - 2*T[1:-1, j] + T[1:-1, j-1]) + 0.5 * dt * Q0 * np.exp(-(x[1:-1]**2 + y[j]**2)/(2*sigma**2))
        d[0] += rx * T_star[0, j]
        d[-1] += rx * T_star[-1, j]
        T_star[1:-1, j] = thomas(a, b, c, d)
    # Apply boundary conditions
    T_star[0, :] = 1.0
    T_star[-1, :] = 1.0
    T_star[:, 0] = 1.0
    T_star[:, -1] = 1.0

    # Second half-step (y-direction implicit)
    T_new = T_star.copy()
    for i in range(1, nx-1):
        a = -ry * np.ones(ny-2)
        b = (1 + 2*ry) * np.ones(ny-2)
        c = -ry * np.ones(ny-2)
        d = T_star[i, 1:-1] + rx * (T_star[i+1, 1:-1] - 2*T_star[i, 1:-1] + T_star[i-1, 1:-1]) + 0.5 * dt * Q0 * np.exp(-(x[i]**2 + y[1:-1]**2)/(2*sigma**2))
        d[0] += ry * T_new[i, 0]
        d[-1] += ry * T_new[i, -1]
        T_new[i, 1:-1] = thomas(a, b, c, d)
    # Apply boundary conditions
    T_new[0, :] = 1.0
    T_new[-1, :] = 1.0
    T_new[:, 0] = 1.0
    T_new[:, -1] = 1.0

    T = T_new.copy()

# Save final temperature field
save_values = ['T']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/T_2D_Unsteady_Heat_Equation_ADI.npy', T)