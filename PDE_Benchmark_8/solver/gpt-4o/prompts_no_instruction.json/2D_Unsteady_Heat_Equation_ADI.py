import numpy as np

# Parameters
alpha = 0.01  # thermal diffusivity
Q0 = 200.0  # source term strength
sigma = 0.1  # source term spread
nx, ny = 41, 41  # grid resolution
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
r = 0.25  # stability parameter
dt = r * dx**2 / alpha
t_max = 3.0
nt = int(t_max / dt)

# Initial condition
T = np.zeros((nx, ny))

# Source term
X, Y = np.meshgrid(x, y, indexing='ij')
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# ADI method
def adi_step(T, q, alpha, dt, dx, dy):
    # Half step in x-direction
    T_half = np.copy(T)
    for j in range(1, ny-1):
        A = np.zeros((nx, nx))
        B = np.zeros(nx)
        for i in range(1, nx-1):
            A[i, i-1] = -alpha * dt / (2 * dx**2)
            A[i, i] = 1 + alpha * dt / dx**2
            A[i, i+1] = -alpha * dt / (2 * dx**2)
            B[i] = T[i, j] + dt * q[i, j] / 2
        A[0, 0] = A[-1, -1] = 1  # Dirichlet boundary conditions
        T_half[:, j] = np.linalg.solve(A, B)

    # Half step in y-direction
    T_new = np.copy(T_half)
    for i in range(1, nx-1):
        A = np.zeros((ny, ny))
        B = np.zeros(ny)
        for j in range(1, ny-1):
            A[j, j-1] = -alpha * dt / (2 * dy**2)
            A[j, j] = 1 + alpha * dt / dy**2
            A[j, j+1] = -alpha * dt / (2 * dy**2)
            B[j] = T_half[i, j] + dt * q[i, j] / 2
        A[0, 0] = A[-1, -1] = 1  # Dirichlet boundary conditions
        T_new[i, :] = np.linalg.solve(A, B)

    return T_new

# Time-stepping loop
for n in range(nt):
    T = adi_step(T, q, alpha, dt, dx, dy)

# Save the final solution
np.save('final_temperature.npy', T)