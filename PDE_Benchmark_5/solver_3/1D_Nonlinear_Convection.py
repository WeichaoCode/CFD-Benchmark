import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
L = 2.0   # length of the domain
T = 1.0   # time of simulation
nx = 101  # number of grid points in space
nt = 101  # number of time steps
dx = L / (nx - 1)  # spatial grid size
dt = T / (nt - 1)  # time step size
cfl = 0.5  # Courant-Friedrichs-Lewy (CFL) number

# Ensure stability via the CFL condition
dt = min(dt, cfl*dx)

# Step 2: Discretize space and time (grid)
x = np.linspace(0, L, nx)
u = np.zeros(nx)  # solution array

# Step 3: Set up the initial condition
u = np.where((0.5 <= x) & (x <= 1.0), 2.0, 1.0)  # initial condition
u_initial = np.copy(u)  # save the initial condition

# Step 4: Finite Difference Scheme Iteration
for n in range(nt):
    u_new = np.copy(u)
    u_new[1:] = u[1:] - dt * u[1:] * (u[1:] - u[:-1]) / dx  # First order upwind scheme
    u = u_new

# Step 5: Plot solution
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, u_initial, label="Initial")
ax.plot(x, u, label="Numerical")
ax.set_xlabel('Space (x)')
ax.set_ylabel('Wave Amplitude (u)')
ax.legend()
plt.show()