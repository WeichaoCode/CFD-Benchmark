import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.5
dt = 0.01
T = 500
dx = dt / nu
x = np.arange(0, 2*np.pi, dx)
u = np.sin(x) + 0.5*np.sin(0.5*x)  # initial condition

# Lax-Wendroff method
for n in range(T):
    un = u.copy()
    F = 0.5*un**2  # Flux
    A = un  # A = u for our PDE
    u[1:-1] = un[1:-1] - dt/(2*dx)*(F[2:] - F[:-2]) + dt**2/(2*dx**2)*((A[2:]+A[1:-1])*(F[2:]-F[1:-1]) - (A[1:-1]+A[:-2])*(F[1:-1]-F[:-2]))
    
    # Periodic boundary conditions
    u[0] = un[0] - dt/(2*dx)*(F[1] - F[-1]) + dt**2/(2*dx**2)*((A[1]+A[0])*(F[1]-F[0]) - (A[0]+A[-1])*(F[0]-F[-1]))
    u[-1] = un[-1] - dt/(2*dx)*(F[0] - F[-2]) + dt**2/(2*dx**2)*((A[0]+A[-1])*(F[0]-F[-1]) - (A[-1]+A[-2])*(F[-1]-F[-2]))

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/u_1D_Nonlinear_Convection_LW.npy', u)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(x, u, label='Lax-Wendroff')
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of the 1D Nonlinear Convection Equation')
plt.grid(True)
plt.show()