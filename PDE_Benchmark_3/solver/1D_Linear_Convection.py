import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, exp, sin

# Step 1: Define Parameters
nx = 81          # number of x grid points
nt = 800         # number of time steps
c  = 1.0         # wave speed
L  = 1.0         # size of domain
dx = L / (nx-1)  # spatial resolution
T  = 0.6         # final time
dt = T / nt      # time step size

# Step 2: Check CFL condition
CFL = c * dt / dx  
assert CFL <= 1.0, "CFL condition not met."

# Step 3: Compute the source term
x = np.linspace(0, L, nx)
t = np.arange(nt) * dt
F = -exp(-t) * pi * sin(pi*x) - c * exp(-t) * pi * cos(pi*x)  # source term

# Step 4: Compute the initial and boundary conditions
u0 = sin(pi * x)      # initial condition
ub = 0                # boundary conditions

# Step 5: Solve the PDE with finite differences
u = np.zeros((nt, nx))
u[0, :] = u0
u[:, (0, -1)] = ub
for n in range(nt - 1):
    u[n+1, 1:-1] = u[n, 1:-1] - c * dt / (2 * dx) * (u[n, 2:] - u[n, :-2]) + dt * F[n, 1:-1]

# Step 6: Compute exact solution
ue = np.outer(exp(-t), sin(pi*x))

# Step 7: Error analysis and Plotting
error = np.abs(u - ue)
print(f"Max error: {np.max(error):.5f}")
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.pcolormesh(u, cmap='viridis')
plt.title("Numeric Solution")
plt.subplot(1, 2, 2)
plt.pcolormesh(ue, cmap='viridis')
plt.title("Exact Solution")
plt.show()