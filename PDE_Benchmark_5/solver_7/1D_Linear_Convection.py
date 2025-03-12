import numpy as np
import matplotlib.pyplot as plt

# 1. Define parameters
L = 5.0  # length of the domain
T = 2.0  # time of the simulation
nx = 101  # number of spatial points
nt = 51  # number of time steps
c = 1.0  # wave speed

# 2. Discretize space and time
dx = L / (nx - 1)  # spatial resolution
dt = T / (nt - 1)  # time resolution

# check the CFL condition
if c*dt/dx > 1:
    print("Warning: solution may be unstable due to failure of CFL condition")

# Create the grid
x = np.linspace(0, L, nx)
u = np.zeros((nt, nx))

# 3. Set the initial condition
u_initial = np.sin(2.0 * np.pi * x / L)
u[0, :] = u_initial

# 4. Finite difference scheme
for t in range(1, nt):
    for i in range(1, nx-1):
        u[t, i] = u[t-1, i] - dt/(2*dx) * c * (u[t-1, i+1] - u[t-1, i-1])

# 5. Plot and visualize data
for t in range(nt):
    if t % 10 == 0:  # only print every 10th frame
        plt.plot(x, u_initial, 'k-')
        plt.plot(x, u[t, :], 'b-')
        plt.title("Time step {:0>2}".format(t))
        plt.ylim([u.min()-1, u.max()+1])
        plt.xlabel('x')
        plt.ylabel('u')
        plt.show()