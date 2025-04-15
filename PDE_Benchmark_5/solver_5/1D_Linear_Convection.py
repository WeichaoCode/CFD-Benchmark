import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 2.0   # length of the domain
T = 1.0   # time of simulation
nx = 100  # number of spatial points
nt = 100  # number of time steps
c = 1.0   # wave speed

# Discretize space and time
dx = L / (nx - 1)
dt = T / (nt - 1)

# Ensure CFL stability condition
if c * dt / dx > 1:
    raise ValueError("The simulation is unstable due to the CFL condition not satisfied!")

# Set up the initial wave profile
u = np.ones(nx)
u[int(.5 / dx):int(1 / dx + 1)] = 2  # set u = 2 between 0.5 and 1

# Prepare for plotting
plt.figure(figsize=(7, 5))

nr_plots = 6
plot_step = nt // nr_plots

# Iterate using the finite difference scheme
for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        u[i] = un[i] - c * dt / (2 * dx) * (un[i + 1] - un[i - 1])

    if n % plot_step == 0:
        plt.plot(np.linspace(0, L, nx), u)

# Plot the wave evolution
plt.ylim([1., 2.2])
plt.xlabel('Space')
plt.ylabel('Wave Amplitude')
plt.show()