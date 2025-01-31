import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0 / (nx - 1)  # spatial step size
dt = 0.002  # time step size (chosen for stability)
c = 1.0  # wave speed

# Grid points
x = np.linspace(0, 2, nx)
t = np.linspace(0, 2, nt)

# Initialize solution array
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# FTCS scheme
def source_term(x, t):
    return -np.pi * c * np.exp(-t) * np.cos(np.pi * x) + np.exp(-t) * np.sin(np.pi * x)

# Check stability (von Neumann analysis)
CFL = c * dt / dx
print(f"CFL number: {CFL}")
if CFL > 1:
    print("Warning: Scheme might be unstable!")

# Solve using FTCS
for n in range(0, nt-1):
    for i in range(1, nx-1):
        u[n+1, i] = u[n, i] - 0.5 * c * dt/dx * (u[n, i+1] - u[n, i-1]) + dt * source_term(x[i], t[n])
    
    # Apply boundary conditions
    u[n+1, 0] = 0
    u[n+1, -1] = 0

# Plot results
plt.figure(figsize=(10, 6))
time_steps = [0, nt//4, nt//2, nt-1]
labels = ['t = 0', 't = T/4', 't = T/2', 't = T']

for i, n in enumerate(time_steps):
    plt.plot(x, u[n, :], label=labels[i])

plt.title('1D Linear Convection - FTCS Method')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Calculate and print maximum error at final time
# (assuming we know analytical solution)
def analytical_solution(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

error = np.max(np.abs(u[-1, :] - analytical_solution(x, 2)))
print(f"Maximum error at t = T: {error}")