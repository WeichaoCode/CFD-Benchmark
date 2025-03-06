import numpy as np
import matplotlib.pyplot as plt

# Exact Solution (Manufactured Solution)
def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

# derivative of the exact solution with respect to x
def dudx_exact(x, t):
    return np.pi * np.exp(-t) * np.cos(np.pi * x)

# derivative of the exact solution with respect to t
def dudt_exact(x, t):
    return -np.exp(-t) * np.sin(np.pi * x)

# source term for the PDE
def source_term(x, t):
    return dudt_exact(x, t) + u_exact(x, t) * dudx_exact(x, t)


# Solver parameters
x_start, x_end = 0, 1  # spatial domain
t_start, t_end = 0, 1  # time domain
nx = 101  # number of spatial points
nt = 100  # number of time steps
x = np.linspace(x_start, x_end, nx)  # spatial grid points
t = np.linspace(t_start, t_end, nt)  # time grid points
dt = t[1] - t[0]  # time step size
dx = x[1] - x[0]  # spatial grid size

# Initialize solution (including boundary)
u = np.empty((nt, nx))
u[:, :] = u_exact(x, t_start)  # initial condition

# Time-stepping loop
for n in range(nt-1):
    # Compute interior points using forward differencing for the convective term
    for i in range(1, nx-1):
        # Enforce Courant number ≤ 0.5 for stability with explicit scheme
        dt = min(dt, 0.5 * dx / np.abs(u[n, i]))
        
        u[n+1, i] = u[n, i] - dt / dx * u[n, i] * (u[n, i] - u[n, i - 1]) + dt * source_term(x[i], t[n])
        
    # Update boundary conditions (from MMS)
    u[n+1, 0] = u_exact(x_start, t[n+1])  # left
    u[n+1, -1] = u_exact(x_end, t[n+1])  # right

# Output
plt.figure(figsize=(9, 6))

# Plot numerical solution
plt.subplot(2, 1, 1) 
plt.imshow(u, extent=[x_start, x_end, t_end, t_start], aspect='auto')
plt.colorbar()
plt.title('Numerical solution')

# Plot error
errors = np.abs(u - u_exact(x, t[:, None]))
plt.subplot(2, 1, 2)
plt.imshow(errors, extent=[x_start, x_end, t_end, t_start], aspect='auto')
plt.colorbar()
plt.title('Absolute error')

plt.tight_layout()
plt.show()