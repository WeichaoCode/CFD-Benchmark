import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.3  # viscosity
L = 2.0   # length of domain
T = 2.0   # total time
nx = 100  # number of spatial points
dx = L/(nx-1)

# Calculate dt based on stability condition for FTCS
# For FTCS, we need dt <= dx^2/(2*nu) for stability
dt = 0.8 * dx**2/(2*nu)  # using safety factor of 0.8
nt = int(T/dt)  # number of time steps

# Create spatial and temporal grids
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Initialize solution array
u = np.zeros((nt, nx))

# Set initial condition
u[0,:] = np.sin(np.pi*x)

# FTCS scheme
def source_term(x, t):
    return -np.pi**2 * nu * np.exp(-t) * np.sin(np.pi*x) + np.exp(-t) * np.sin(np.pi*x)

# Time stepping
for n in range(0, nt-1):
    for i in range(1, nx-1):
        u[n+1,i] = u[n,i] + nu*dt/(dx**2) * (u[n,i+1] - 2*u[n,i] + u[n,i-1]) + \
                   dt * source_term(x[i], t[n])
    
    # Apply boundary conditions
    u[n+1,0] = 0
    u[n+1,-1] = 0

# Plot results at specified time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0,:], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4),:], 'r--', label=f't = {T/4:.2f}')
plt.plot(x, u[int(nt/2),:], 'g-.', label=f't = {T/2:.2f}')
plt.plot(x, u[-1,:], 'k:', label=f't = {T:.2f}')

plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('1D Diffusion Equation - FTCS Method')
plt.legend()
plt.grid(True)
plt.show()

# Print stability parameters
print(f"dx = {dx:.6f}")
print(f"dt = {dt:.6f}")
print(f"Stability parameter (nu*dt/dx^2) = {nu*dt/dx**2:.6f}")