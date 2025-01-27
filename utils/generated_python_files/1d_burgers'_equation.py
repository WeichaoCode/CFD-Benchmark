import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nu = 0.07  # Viscosity
Lx = 2 * np.pi  # Domain length
Nx = 101  # Number of spatial points
dx = Lx / (Nx - 1)  # Grid spacing
sigma = 0.2  # CFL-like parameter for stability
dt = sigma * dx**2 / nu  # Time step
Nt = 200  # Number of time steps

# Define spatial grid
x = np.linspace(0, Lx, Nx)

# Define initial condition
phi = np.exp(-x**2 / (4 * nu)) + np.exp(-(x - 2*np.pi)**2 / (4 * nu))
dphi_dx = (-x / (4 * nu) * np.exp(-x**2 / (4 * nu))) + (-(x - 2*np.pi) / (4 * nu) * np.exp(-(x - 2*np.pi)**2 / (4 * nu)))
u = -2 * nu * dphi_dx / phi + 4  # Compute initial velocity

# Time stepping loop
for n in range(Nt):
    u_new = u.copy()  # Copy current state for update
    
    # First derivative (convection term) using upwind scheme
    du_dx = (u - np.roll(u, 1)) / dx
    
    # Second derivative (diffusion term) using central difference
    d2u_dx2 = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
    
    # Update equation (Explicit scheme)
    u_new = u - dt * u * du_dx + nu * dt * d2u_dx2

    # Enforce periodic boundary conditions
    u = u_new.copy()

# Plot results
plt.plot(x, u, label=f"t = {Nt * dt:.2f}")
plt.xlabel("x")
plt.ylabel("u")
plt.title("1D Burgers' Equation Solution")
plt.legend()
plt.grid()
plt.show()

