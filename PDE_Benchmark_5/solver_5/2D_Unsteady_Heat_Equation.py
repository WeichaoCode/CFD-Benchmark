import numpy as np
import matplotlib.pyplot as plt

# Define constants
alpha = 0.0001
Q_0 = 200.0
sigma = 0.1
Lx = 2
Ly = 2
Nx = 100
Ny = 100
dx = Lx / Nx
dy = Ly / Ny

# Define time parameters
dt = 0.01
t_end = 1.0

# Define initial condition T and source term Q
T = np.zeros((Nx, Ny))
X, Y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))
Q = Q_0 * np.exp(-(X**2 + Y**2) / (2*sigma**2))

# Implicit method solver for heat equation
def solver_implicit(T, Q, alpha, dt, dx, dy):
    # insert code functionality here,
    return T

# Explicit method solver for heat equation
def solver_explicit(T, Q, alpha, dt, dx, dy):
    # insert code functionality here
    return T 

time = 0.0
while time < t_end:
    T_implicit = solver_implicit(T.copy(), Q, alpha, dt, dx, dy)
    T_explicit = solver_explicit(T.copy(), Q, alpha, dt, dx, dy)
    T = T_implicit.copy()
    time += dt

# plot the solution  
plt.figure()   
plt.contourf(X, Y, T, cmap='hot')
plt.title('Temperature Distribution')
plt.colorbar(label='Temperature (°C)')
plt.show()