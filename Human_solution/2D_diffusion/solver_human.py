import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nx, ny = 50, 50  # Grid points
nt = 200  # Time steps
T = 0.2  # Final time
Lx, Ly = 1.0, 1.0  # Domain size
nu = 0.05

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = T / (nt - 1)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.linspace(0, T, nt)


# source term
def f(x_f, y_f, t_f):
    return (2 * nu * np.pi ** 2 - 1) * np.exp(-t_f) * np.sin(np.pi * x_f) * np.sin(np.pi * y_f)


# set the initial conditions
u = np.zeros((nx, ny, nt))

for i in range(nx):
    for j in range(ny):
        u[i, j, 0] = np.exp(-t[0]) * np.sin(np.pi * x[i]) * np.sin(np.pi * y[j])

# set the boundary condition x = 0
for j in range(ny):
    for n in range(nt):
        u[0, j, n] = np.exp(-t[n]) * np.sin(np.pi * x[0]) * np.cos(np.pi * y[j])

# set the boundary condition y = 0
for i in range(nx):
    for n in range(nt):
        u[i, 0, n] = np.exp(-t[n]) * np.sin(np.pi * x[i]) * np.cos(np.pi * y[0])

# set the boundary condition x = Lx
for j in range(ny):
    for n in range(nt):
        u[-1, j, n] = np.exp(-t[n]) * np.sin(np.pi * x[-1]) * np.cos(np.pi * y[j])

# set the boundary condition y = Ly
for i in range(nx):
    for n in range(nt):
        u[i, -1, n] = np.exp(-t[n]) * np.sin(np.pi * x[i]) * np.cos(np.pi * y[-1])
# main loop
for n in range(nt - 1):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u[i, j, n + 1] = (u[i, j, n] + nu * dt / dx ** 2 * (u[i + 1, j, n] - 2 * u[i - 1, j, n] + u[i - 1, j, n])
                              + nu * dt / dy ** 2 * (u[i, j + 1, n]) - 2 * u[i, j, n] + u[i, j - 1, n]
                              + dt * f(x[i], y[j], t[n]))

# compute the exact solution
u_exact = np.zeros((nx, ny, nt))
for n in range(nt):
    for i in range(nx):
        for j in range(ny):
            u_exact[i, j, n] = np.exp(-t[n]) * np.sin(np.pi * x[i]) * np.cos(np.pi * y[j])

# plot the results
X, Y = np.meshgrid(x, y, indexing="ij")  # Generate meshgrid for plotting

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Numerical solution
ax1 = axes[0]
c1 = ax1.contourf(X, Y, u[..., -1], cmap="viridis")
plt.colorbar(c1, ax=ax1)
ax1.set_title("Numerical Solution")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# MMS (exact) solution
ax2 = axes[1]
c2 = ax2.contourf(X, Y, u_exact[..., -1], cmap="viridis")
plt.colorbar(c2, ax=ax2)
ax2.set_title("MMS (Exact) Solution")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Absolute Error Plot
error = u[..., -1] - u_exact[..., -1]
ax3 = axes[2]
c3 = ax3.contourf(X, Y, error, cmap="inferno")  # Using 'inferno' to highlight errors
plt.colorbar(c3, ax=ax3)
ax3.set_title("Absolute Error |u - MMS|")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

plt.suptitle(f"Comparison at t = {T}", fontsize=14)
plt.tight_layout()
plt.show()