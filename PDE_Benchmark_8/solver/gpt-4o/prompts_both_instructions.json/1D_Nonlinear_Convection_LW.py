import numpy as np

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
T = 500  # number of time steps

# Discretize the spatial domain
x = np.linspace(0, L, int(np.ceil(L / dx)), endpoint=False)
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Lax-Wendroff method
for n in range(T):
    # Compute flux
    F = 0.5 * u**2
    
    # Compute A at half indices
    A_plus_half = (u + np.roll(u, -1)) / 2
    A_minus_half = (u + np.roll(u, 1)) / 2
    
    # Update u using Lax-Wendroff scheme
    u = (u - (dt / (2 * dx)) * (np.roll(F, -1) - np.roll(F, 1)) +
         (dt**2 / (2 * dx**2)) * (A_plus_half * (np.roll(F, -1) - F) -
                                  A_minus_half * (F - np.roll(F, 1))))
    
    # Apply periodic boundary conditions
    u[0] = u[-1]

# Save the final solution
np.save('final_solution.npy', u)