# CFD Solver for 1D Euler equations Close Form Differentiation
import numpy as np
import matplotlib.pyplot as plt

# Physical properties
gamma = 1.4 # Perfect Gas

# Grid parameters
nx = 100  # number of grid points
nt = 250  # number of time steps
dt = 0.0001 # timestep size

# Mapping from conserved variables to primitives
def cons2prim(Q):
    u = Q[1] / Q[0]
    p = (gamma - 1.0) * (Q[2] - 0.5 * Q[1]**2 / Q[0])
    return np.array([Q[0], u, p])


# Flux function definition
def flux(Q):
    rho, u, p = cons2prim(Q)
    return np.array([
        Q[1],
        Q[1] ** 2 / Q[0] + p,
        (Q[2] + p) * u
    ])

# Lax-Friedrichs Flux
def LF(Ql, Qr, dx, dt):
    return 0.5 * (flux(Ql) + flux(Qr) - dx / dt * (Qr - Ql))


# Simulation
x = np.linspace(0.0, 1.0, nx)  # x-coordinate
Q = np.empty((3, nx))  # container for conserved variables
Q[0] = np.exp(-x) * np.sin(np.pi * x)  # rho
Q[1] = np.exp(-x) * Q[0] * np.cos(np.pi * x)  # m
Q[2] = np.exp(-x) * (1 + np.sin(np.pi * x))  # E

# Initialize results vector at Q_new
Q_new = np.empty_like(Q)
dx = 1.0/nx #Grid size calculation

# Compute numerical solution
for n in range(nt):
  
    # Loop over the grid points and update Q for each
    for i in range(1, nx - 1):
        Q_new[:, i] =(Q[:, i] - 0.5 * dt / dx * 
                     (flux(Q[:, i + 1]) - flux(Q[:, i - 1])) +
                     dt / dx ** 2 * (Q[:, i - 1] - 2 * Q[:, i] + Q[:, i + 1]))
    #Update Q with Q_new    
    Q = Q_new.copy()

#Data visualization
plt.figure(figsize=(9, 4), dpi=100)
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(x, Q[i,:])
    plt.gca().set_title(Q[i])

plt.tight_layout()
plt.show()