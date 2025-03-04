import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ---------------- PARAMETERS ----------------
Lx, Ly = 2.0, 2.0  # Domain size
nu = 0.1           # Kinematic viscosity
nt = 500           # Number of time steps
nx, ny = 41, 41    # Grid points
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = 0.001         # Time step

# Stability check (CFL condition)
cfl = dt * max(1/dx, 1/dy)
if cfl >= 0.5:
    raise ValueError(f"CFL condition violated: cfl = {cfl:.2f}")

# Define spatial and temporal grids
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# ---------------- MMS Solution ----------------
def mms_u(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

def mms_v(x, y, t):
    return np.exp(-t) * np.cos(np.pi * x) * np.cos(np.pi * y)

# ---------------- Source Terms from MMS ----------------
def f_u(x, y, t):
    return (-np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y) +
            np.exp(-2 * t) * (np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) -
                              np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)) -
            nu * np.pi ** 2 * np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y))

def f_v(x, y, t):
    return (-np.exp(-t) * np.cos(np.pi * x) * np.cos(np.pi * y) +
            np.exp(-2 * t) * (-np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) +
                              np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)) -
            nu * np.pi ** 2 * np.exp(-t) * np.cos(np.pi * x) * np.cos(np.pi * y))

# ---------------- INITIAL CONDITIONS FROM MMS ----------------
u = mms_u(X, Y, 0)
v = mms_v(X, Y, 0)

# ---------------- SOLVE USING FINITE DIFFERENCE ----------------
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Compute source terms
    f_u_t = f_u(X, Y, n * dt)
    f_v_t = f_v(X, Y, n * dt)

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u[i, j] = (un[i, j] -
                       dt / dx * un[i, j] * (un[i, j] - un[i-1, j]) -
                       dt / dy * vn[i, j] * (un[i, j] - un[i, j-1]) +
                       nu * dt / dx**2 * (un[i+1, j] - 2 * un[i, j] + un[i-1, j]) +
                       nu * dt / dy**2 * (un[i, j+1] - 2 * un[i, j] + un[i, j-1]) +
                       dt * f_u_t[i, j])

            v[i, j] = (vn[i, j] -
                       dt / dx * un[i, j] * (vn[i, j] - vn[i-1, j]) -
                       dt / dy * vn[i, j] * (vn[i, j] - vn[i, j-1]) +
                       nu * dt / dx**2 * (vn[i+1, j] - 2 * vn[i, j] + vn[i-1, j]) +
                       nu * dt / dy**2 * (vn[i, j+1] - 2 * vn[i, j] + vn[i, j-1]) +
                       dt * f_v_t[i, j])

    # Apply MMS boundary conditions at next time step
    u[0, :] = mms_u(x[0], y, (n + 1) * dt)  # Left boundary
    u[-1, :] = mms_u(x[-1], y, (n + 1) * dt)  # Right boundary
    u[:, 0] = mms_u(x, y[0], (n + 1) * dt)  # Bottom boundary
    u[:, -1] = mms_u(x, y[-1], (n + 1) * dt)  # Top boundary

    v[0, :] = mms_v(x[0], y, (n + 1) * dt)
    v[-1, :] = mms_v(x[-1], y, (n + 1) * dt)
    v[:, 0] = mms_v(x, y[0], (n + 1) * dt)
    v[:, -1] = mms_v(x, y[-1], (n + 1) * dt)

# ---------------- COMPUTE EXACT SOLUTION FOR COMPARISON ----------------
u_exact = mms_u(X, Y, nt * dt)
v_exact = mms_v(X, Y, nt * dt)

# ---------------- ERROR ANALYSIS ----------------
error_u = np.abs(u - u_exact)
error_v = np.abs(v - v_exact)

# ---------------- PLOTTING RESULTS ----------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Numerical solution for u at final time step
ax1 = axes[0, 0]
c1 = ax1.contourf(X, Y, u, cmap="jet")
plt.colorbar(c1, ax=ax1)
ax1.set_title("Numerical Solution U at t = T")

# MMS (exact) solution for u
ax2 = axes[0, 1]
c2 = ax2.contourf(X, Y, u_exact, cmap="jet")
plt.colorbar(c2, ax=ax2)
ax2.set_title("MMS (Exact) Solution U at t = T")

# Absolute Error in u
ax3 = axes[0, 2]
c3 = ax3.contourf(X, Y, error_u, cmap="inferno")
plt.colorbar(c3, ax=ax3)
ax3.set_title("Absolute Error |U - MMS| at t = T")

# Numerical solution for v at final time step
ax4 = axes[1, 0]
c4 = ax4.contourf(X, Y, v, cmap="jet")
plt.colorbar(c4, ax=ax4)
ax4.set_title("Numerical Solution V at t = T")

# MMS (exact) solution for v
ax5 = axes[1, 1]
c5 = ax5.contourf(X, Y, v_exact, cmap="jet")
plt.colorbar(c5, ax=ax5)
ax5.set_title("MMS (Exact) Solution V at t = T")

# Absolute Error in v
ax6 = axes[1, 2]
c6 = ax6.contourf(X, Y, error_v, cmap="inferno")
plt.colorbar(c6, ax=ax6)
ax6.set_title("Absolute Error |V - MMS| at t = T")

plt.suptitle("Comparison of Solutions and Errors at Final Time", fontsize=14)
plt.tight_layout()
plt.savefig("Burgers_MMS_Comparison.png", bbox_inches="tight")
plt.show()





