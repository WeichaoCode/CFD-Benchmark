import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 1.0  # length of the domain
T = 0.5  # time interval
nx = 100  # number of spatial points in grid
nt = 50  # number of time steps
dx = L/nx  # spatial resolution
dt = T/nt  # temporal resolution
cfl = dt/dx  # CFL condition

# Discretize the domain
x = np.linspace(0, L, nx)

# Set up initial wave profile
u0 = np.where((0.4 <= x) & (x <= 0.6), 1.0, 0.0)  # initially, u = 1 between 0.4 and 0.6, u = 0 elsewhere
u = u0.copy()

# Set up figure
fig = plt.figure(figsize=(6,3), dpi=100)
ax = fig.add_subplot(1,1,1)
line, = ax.plot(x, u)

# Finite difference calculation and plot
for t in range(1,nt):
    un = u.copy()
    for i in range(1,nx):
        u[i] = un[i] - un[i]*dt/dx*(un[i] - un[i-1])
    line.set_ydata(u)
    ax.set_title('Time: {0:.2f}'.format(t*dt))
    plt.draw()
    plt.pause(0.1)

# Show the final figure
plt.show()