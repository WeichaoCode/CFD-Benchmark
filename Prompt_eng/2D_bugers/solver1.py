import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ---------------- PARAMETERS ----------------
Lx, Ly = 1.0, 1.0  # Domain size
T = 2.0  # Final time
ν = 0.05  # Kinematic viscosity

# Discretization
nx, ny = 50, 50  # Grid points
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = 0.001  # Time step (chosen for stability)

# Stability check (CFL condition for convection-dominated problems)
dt_max = min(dx, dy) / np.sqrt(2)
if dt > dt_max:
    print(f"Reducing dt from {dt:.6f} to {dt_max:.6f} for stability.")
    dt = dt_max

# Define spatial and temporal grids
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.arange(0, T + dt, dt)
X, Y = np.meshgrid(x, y, indexing='ij')


# ---------------- MMS Solution ----------------
def mms_u(x_f, y_f, t_f):
    return np.exp(-t_f) * np.sin(np.pi * x_f) * np.sin(np.pi * y_f)


def mms_v(x_f, y_f, t_f):
    return np.exp(-t_f) * np.cos(np.pi * x_f) * np.cos(np.pi * y_f)


# ---------------- Source Terms from MMS ----------------
def f_u(x_f, y_f, t_f):
    return (-np.exp(-t_f) * np.sin(np.pi * x_f) * np.sin(np.pi * y_f) +
            np.exp(-2 * t_f) * (np.pi * np.sin(np.pi * x_f) * np.cos(np.pi * y_f) -
                                np.pi * np.cos(np.pi * x_f) * np.sin(np.pi * y_f)) -
            ν * np.pi ** 2 * np.exp(-t_f) * np.sin(np.pi * x_f) * np.sin(np.pi * y_f))


def f_v(x_f, y_f, t_f):
    return (-np.exp(-t_f) * np.cos(np.pi * x_f) * np.cos(np.pi * y_f) +
            np.exp(-2 * t_f) * (-np.pi * np.cos(np.pi * x_f) * np.sin(np.pi * y_f) +
                                np.pi * np.sin(np.pi * x_f) * np.cos(np.pi * y_f)) -
            ν * np.pi ** 2 * np.exp(-t_f) * np.cos(np.pi * x_f) * np.cos(np.pi * y_f))


# ---------------- INITIAL CONDITIONS FROM MMS ----------------
u = np.zeros((nx, ny, len(t)))
v = np.zeros((nx, ny, len(t)))
u[:, :, 0] = mms_u(X, Y, 0)
v[:, :, 0] = mms_v(X, Y, 0)

# ---------------- SOLVE USING FINITE DIFFERENCE (Implicit Backward Euler) ----------------
for n in range(len(t) - 1):
    f_u_t = f_u(X, Y, t[n])
    f_v_t = f_v(X, Y, t[n])

    # Construct implicit matrix for diffusion term
    αx = ν * dt / dx ** 2
    αy = ν * dt / dy ** 2
    main_diag = (1 + 2 * αx + 2 * αy) * np.ones(nx * ny)
    off_diag_x = -αx * np.ones(nx * ny - 1)
    off_diag_y = -αy * np.ones(nx * ny - nx)

    # Apply boundary conditions (Dirichlet from MMS)
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                main_diag[i * ny + j] = 1
                off_diag_x[i * ny + j - 1] = 0
                off_diag_y[i * ny + j - nx] = 0

    # Assemble sparse matrix
    A = diags([off_diag_y, off_diag_x, main_diag, off_diag_x, off_diag_y], [-nx, -1, 0, 1, nx], format="csr")

    # Solve for u and v using implicit scheme
    u_old = u[:, :, n].flatten()
    v_old = v[:, :, n].flatten()

    b_u = u_old + dt * f_u_t.flatten()
    b_v = v_old + dt * f_v_t.flatten()

    u_new = spsolve(A, b_u).reshape((nx, ny))
    v_new = spsolve(A, b_v).reshape((nx, ny))

    # Apply MMS boundary conditions at the next time step
    u_new[0, :] = mms_u(x[0], y, t[n + 1])  # Left boundary
    u_new[-1, :] = mms_u(x[-1], y, t[n + 1])  # Right boundary
    u_new[:, 0] = mms_u(x, y[0], t[n + 1])  # Bottom boundary
    u_new[:, -1] = mms_u(x, y[-1], t[n + 1])  # Top boundary

    v_new[0, :] = mms_v(x[0], y, t[n + 1])
    v_new[-1, :] = mms_v(x[-1], y, t[n + 1])
    v_new[:, 0] = mms_v(x, y[0], t[n + 1])
    v_new[:, -1] = mms_v(x, y[-1], t[n + 1])

    # Store the updated solution
    u[:, :, n + 1] = u_new
    v[:, :, n + 1] = v_new

# ---------------- COMPUTE EXACT SOLUTION FOR COMPARISON ----------------
u_exact = np.zeros((nx, ny, len(t)))
v_exact = np.zeros((nx, ny, len(t)))

for n in range(len(t)):
    u_exact[:, :, n] = mms_u(X, Y, t[n])
    v_exact[:, :, n] = mms_v(X, Y, t[n])

# ---------------- ERROR ANALYSIS ----------------
error_u = np.abs(u - u_exact)
error_v = np.abs(v - v_exact)

# ---------------- PLOTTING RESULTS ----------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Numerical solution for u at final time step
ax1 = axes[0, 0]
c1 = ax1.contourf(X, Y, u[:, :, -1], cmap="viridis")
plt.colorbar(c1, ax=ax1)
ax1.set_title("Numerical Solution u at t = T")

# MMS (exact) solution for u
ax2 = axes[0, 1]
c2 = ax2.contourf(X, Y, u_exact[:, :, -1], cmap="viridis")
plt.colorbar(c2, ax=ax2)
ax2.set_title("MMS (Exact) Solution u at t = T")

# Absolute Error in u
ax3 = axes[0, 2]
c3 = ax3.contourf(X, Y, error_u[:, :, -1], cmap="inferno")
plt.colorbar(c3, ax=ax3)
ax3.set_title("Absolute Error |u - MMS| at t = T")

# Numerical solution for v at final time step
ax1 = axes[1, 0]
c1 = ax1.contourf(X, Y, v[:, :, -1], cmap="viridis")
plt.colorbar(c1, ax=ax1)
ax1.set_title("Numerical Solution u at t = T")

# MMS (exact) solution for v
ax2 = axes[1, 1]
c2 = ax2.contourf(X, Y, v_exact[:, :, -1], cmap="viridis")
plt.colorbar(c2, ax=ax2)
ax2.set_title("MMS (Exact) Solution u at t = T")

# Absolute Error in v
ax3 = axes[1, 2]
c3 = ax3.contourf(X, Y, error_v[:, :, -1], cmap="inferno")
plt.colorbar(c3, ax=ax3)
ax3.set_title("Absolute Error |v - MMS| at t = T")

plt.suptitle(f"Comparison at t = {T}", fontsize=14)
plt.tight_layout()
plt.savefig("/opt/CFD-Benchmark/Prompt_eng/2D_bugers/output1.png")
plt.show()
