import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.5
dt = 0.01
T = 500
dx = dt / nu
x = np.arange(0, 2*np.pi, dx)
u = np.sin(x) + 0.5*np.sin(0.5*x)  # initial condition
u_new = np.empty_like(u)

# Time loop
for t in range(T):
    F = 0.5 * u**2  # Nonlinear convection term
    u_new[1:-1] = 0.5 * (u[:-2] + u[2:]) - dt / (2*dx) * (F[2:] - F[:-2])
    
    # Periodic boundary conditions
    u_new[0] = 0.5 * (u[-1] + u[1]) - dt / (2*dx) * (F[1] - F[-1])
    u_new[-1] = 0.5 * (u[-2] + u[0]) - dt / (2*dx) * (F[0] - F[-2])
    
    u = u_new.copy()

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/u_1D_Nonlinear_Convection_Lax.npy', u)

# Plot the solution
plt.figure(figsize=(8, 6))
plt.plot(x, u, label='u(x,t)')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of the 1D Nonlinear Convection Equation')
plt.legend()
plt.grid(True)
plt.show()