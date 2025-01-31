import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
c = 1.0  # Wave speed

# Grid parameters
nx = 100  # Number of spatial points
nt = 200  # Number of time points
dx = L / (nx-1)
dt = T / (nt-1)

# Check CFL condition for stability (von Neumann analysis)
CFL = c * dt / dx
print(f"CFL number: {CFL}")
if CFL > 1:
    print("Warning: Solution might be unstable!")

# Initialize grid
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Initialize solution array
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Set boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# First time step using Forward Euler (needed to start Leapfrog)
for i in range(1, nx-1):
    u[1, i] = u[0, i] - 0.5*CFL*(u[0, i+1] - u[0, i-1]) + \
              dt*(np.pi*c*np.exp(-t[0])*np.cos(np.pi*x[i]) - \
                  np.exp(-t[0])*np.sin(np.pi*x[i]))

# Leapfrog time stepping
for n in range(1, nt-1):
    for i in range(1, nx-1):
        source = np.pi*c*np.exp(-t[n])*np.cos(np.pi*x[i]) - \
                 np.exp(-t[n])*np.sin(np.pi*x[i])
        u[n+1, i] = u[n-1, i] - CFL*(u[n, i+1] - u[n, i-1]) + \
                    2*dt*source

# Plot results
plt.figure(figsize=(10, 6))
plot_times = [0, int(nt/4), int(nt/2), nt-1]
labels = ['t = 0', 't = T/4', 't = T/2', 't = T']
for i, n in enumerate(plot_times):
    plt.plot(x, u[n, :], label=labels[i])

plt.title('1D Linear Convection - Leapfrog Method')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Calculate and print maximum error at final time
exact_solution = np.exp(-T) * np.sin(np.pi * x)
error = np.max(np.abs(u[-1, :] - exact_solution))
print(f"Maximum error at t = T: {error}")