import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
nx = 100  # Number of spatial points
nt = 1000  # Number of time steps
dx = L/(nx-1)  # Spatial step size
dt = T/nt  # Time step size

# Initialize arrays
x = np.linspace(0, L, nx)
u = np.zeros((nt+1, nx))

# Initial condition
u[0, :] = np.sin(np.pi*x)

# Boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Source term function
def source(x, t):
    return np.exp(-t)*np.sin(np.pi*x) - np.pi*np.exp(-2*t)*np.sin(np.pi*x)*np.cos(np.pi*x)

# FTCS scheme
for n in range(0, nt):
    for i in range(1, nx-1):
        # Nonlinear convection term using centered difference
        conv = u[n,i]*(u[n,i+1] - u[n,i-1])/(2*dx)
        # Source term
        src = source(x[i], n*dt)
        # Update solution
        u[n+1,i] = u[n,i] - dt*conv + dt*src

# Plot results at specified time steps
plt.figure(figsize=(10, 6))
t_plot = [0, int(nt/4), int(nt/2), nt]
labels = ['t = 0', 't = T/4', 't = T/2', 't = T']
colors = ['b', 'r', 'g', 'k']

for i, t in enumerate(t_plot):
    plt.plot(x, u[t,:], colors[i], label=labels[i])

plt.title('1D Nonlinear Convection - FTCS Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Stability check
c = np.max(np.abs(u))*dt/dx  # CFL number
print(f"CFL number = {c:.3f}")
if c <= 1:
    print("Scheme is stable (CFL condition satisfied)")
else:
    print("Warning: Scheme might be unstable (CFL condition not satisfied)")