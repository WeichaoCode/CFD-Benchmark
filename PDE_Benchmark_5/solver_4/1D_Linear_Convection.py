import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Define parameters
L = 2.0   # length of the domain
T = 1.0   # time of the simulation
nx = 101  # number of spatial points in the grid
nt = 100  # number of time steps
c = 1.0   # wave speed

# Spatial discretization
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

# Time discretization
dt = T / (nt - 1)
t = np.linspace(0, T, nt)

# Ensure stability using the CFL condition
CFL = c * dt / dx
assert CFL <= 1.0, "The CFL condition is not satisfied!"

# Wave initialization
u = np.ones(nx)      # create a u vector of 1's
u[int(.5 / dx):int(1 / dx + 1)] = 2  # then set u = 2 between 0.5 and 1 as per our I.C.s

# Finite Difference Scheme
un = np.ones(nx) # our placeholder array, un, to advance the solution in time

for n in range(nt):  # iterate through time
  un = u.copy() ##copy the existing values of u into un
  for i in range(1, nx - 1): ##now we'll iterate through the u array
    u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])

# Plot
plt.plot(x, u);
plt.xlabel('X'); plt.ylabel('u'); plt.grid(True);
plt.title('1-D Linear Convection');