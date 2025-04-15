import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.5
dt = 0.01
T = 500
dx = dt / nu
x = np.arange(0, 2*np.pi, dx)
u = np.sin(x) + 0.5*np.sin(0.5*x)  # initial condition

# MacCormack method
for t in range(T):
    u_star = u.copy()
    F = 0.5 * u**2  # Flux
    u_star[:-1] = u[:-1] - dt/dx * (F[1:] - F[:-1])
    F_star = 0.5 * u_star**2
    u[1:] = 0.5 * (u[1:] + u_star[1:] - dt/dx * (F_star[1:] - F_star[:-1]))
    u[0] = u[-1]  # periodic boundary condition

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/u_1D_Nonlinear_Convection_Mk.npy', u)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(x, u, label='MacCormack')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of the 1D Nonlinear Convection Equation')
plt.legend()
plt.grid(True)
plt.show()