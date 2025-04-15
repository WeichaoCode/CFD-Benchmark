import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define Parameters
L = 2.0  # length of the domain
T = 0.625  # time of the simulation
nx = 200  # number of spatial points in the domain
nt = 50  # number of time steps
dx = L / (nx - 1)  # spatial discretization size
dt = T / (nt - 1)  # time discretization size

# Step 2: Discretize Space and Time
x = np.linspace(0, L, nx)
u = np.ones(nx)  # initialize u with ones

# Step 3: Set Up Initial Wave Profile
u[int(.5/dx):int(1/dx+1)] = 2  # setting u = 2 between 0.5 and 1 as per our initial condition

# Step 4: Iterate Using FDM
un = np.ones(nx)  # initialize a temporary array

plt.figure(figsize=(11, 7), dpi=100)
plt.plot(np.linspace(0, 2, nx), u)

for n in range(nt):
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i - 1])

# Step 5: Plot the Results
plt.plot(x, u)
plt.ylim([1.,2.2])
plt.xlabel('X')
plt.ylabel('U')
plt.show()