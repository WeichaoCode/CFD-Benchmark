import numpy as np
import matplotlib.pyplot as plt

# Set simulation parameters
L = 1.0   # size of domain
nx, ny = 128, 128  # number of grid points
nt = 1000  # number of time steps
dt = 0.01  # time step size
ν = 0.1  # viscosity
ρ = 1.0  # density 

# Define grid 
dx = L / (nx - 1)
dy = L / (ny - 1)
x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
X, Y = np.meshgrid(x, y)

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

def build_up_b(b, rho, dt, u, v, dx, dy):
    # code for building up b (right-hand side pressure equation)
    return b

def pressure_poisson(p, dx, dy, b):
    # code for Jacobi iteration
    return p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    # code for time-stepping solution
    return u, v, p

u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, ρ, ν)

# Visualization
plt.figure(figsize=(11, 7), dpi=100)
plt.contourf(X, Y, p, alpha=0.5, cmap='viridis')
plt.colorbar()
plt.quiver(X, Y, u, v)  # plot velocity field
plt.show()