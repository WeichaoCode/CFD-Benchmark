import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0  # Convection speed
nx = ny = 81  # Number of grid points
dx = dy = 2.0 / (nx - 1)  # Grid spacing
sigma = 0.2  # CFL number
dt = sigma * min(dx, dy) / c  # Time step size
nt = 100  # Number of time steps

# Initialize the domain
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
u = np.ones((ny, nx))  # Initialize u to 1 everywhere

# Initial condition: hat function
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    # Update u using the upwind scheme
    u[1:, 1:] = (un[1:, 1:] - c * dt / dx * (un[1:, 1:] - un[1:, :-1])
                 - c * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_2D_Linear_Convection.npy', u)

# Plot the final solution
plt.figure(figsize=(8, 6))
plt.contourf(x, y, u, cmap='viridis')
plt.colorbar()
plt.title('Final solution at t = {:.2f}'.format(nt * dt))
plt.xlabel('x')
plt.ylabel('y')
plt.show()