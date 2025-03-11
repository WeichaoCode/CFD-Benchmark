import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
L = 2.0   # length of the domain
T = 1.0   # time of simulation
nx = 101  # number of spatial points in the grid
nt = 100  # number of time steps
c = 1.0   # wave speed

dx = L / (nx-1)
dt = T / (nt-1)
CFL = c * dt / dx

assert CFL <= 1.0, "The CFL condition is not satisfied."

# Step 2: Discretize space and time
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Step 3: Set up the initial wave profile
u0 = np.where((x >= 0.5) & (x <= 1.0), 2, 1)
u = u0.copy()

# Step 4: Iterate using the finite difference scheme
for n in range(nt-1):
    un = u.copy()
    for i in range(1, nx-1):
        u[i] = un[i] - CFL/2 * (un[i+1] - un[i-1])

# Step 5: Plot the wave evolution
plt.figure(figsize=(9, 4))
plt.plot(x, u0, label="Initial")
plt.plot(x, u, label="Final")
plt.legend()
plt.xlim(0, L)
plt.ylim(0, 2.5)
plt.show()