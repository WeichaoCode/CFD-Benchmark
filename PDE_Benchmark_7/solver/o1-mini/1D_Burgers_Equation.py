import numpy as np

# Parameters
nx = 101                           # Number of grid points in x
nt = 100                           # Number of time steps
dx = 2 * np.pi / (nx - 1)          # Spatial step size
nu = 0.07                          # Viscosity coefficient
dt = dx * nu                       # Time step size

# Spatial grid
x = np.linspace(0, 2 * np.pi, nx)

# Initial condition
phi_initial = np.exp(-(x)**2 / (4 * nu)) + np.exp(-(x - 2 * np.pi)**2 / (4 * nu))
dphi_dx_initial = (-x / (2 * nu)) * np.exp(-(x)**2 / (4 * nu)) + (-(x - 2 * np.pi) / (2 * nu)) * np.exp(-(x - 2 * np.pi)**2 / (4 * nu))
u = -2 * nu / phi_initial * dphi_dx_initial + 4

# Time-stepping loop
for _ in range(nt):
    un = u.copy()
    u = un - un * dt / dx * (un - np.roll(un, 1)) \
            + nu * dt / dx**2 * (np.roll(un, -1) - 2 * un + np.roll(un, 1))

# Save the final velocity field to a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_1D_Burgers_Equation.npy', u)