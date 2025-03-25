import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.5          # CFL number
dt = 0.01         # Time step
T = 500           # Maximum number of time steps
dx = dt / nu      # Space step
x = np.arange(0, 2 * np.pi, dx)  # Spatial domain

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Time-stepping loop using the Lax method
for _ in range(T):
    F = 0.5 * u**2  # Flux term
    # Apply periodic boundary conditions using np.roll
    u_plus = np.roll(u, -1)
    u_minus = np.roll(u, 1)
    F_plus = np.roll(F, -1)
    F_minus = np.roll(F, 1)
    # Lax update
    u_new = 0.5 * (u_plus + u_minus) - (dt / (2 * dx)) * (F_plus - F_minus)
    u = u_new

# Plot the final solution
plt.figure(figsize=(8, 4))
plt.plot(x, u, label='u(x, T)')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution at Final Time Step')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the final solution to a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_1D_Nonlinear_Convection_Lax.npy', u)