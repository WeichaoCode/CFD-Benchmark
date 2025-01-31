import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
c = 1.0  # Wave speed
nx = 100  # Number of spatial points
dx = L/(nx-1)  # Spatial step size

# Choose dt based on CFL condition for stability
CFL = 0.8
dt = CFL*dx/abs(c)
nt = int(T/dt)  # Number of time steps

# Grid points
x = np.linspace(0, L, nx)

# Initialize solution array
u = np.zeros(nx)
u_old = np.zeros(nx)

# Initial condition
u = np.sin(np.pi*x)
u[0] = 0  # Boundary condition at x = 0
u[-1] = 0  # Boundary condition at x = 2

# Store initial condition for plotting
u_initial = u.copy()

# Arrays to store solutions at specific time steps
u_quarter = None
u_half = None
u_final = None

# Time stepping
for n in range(nt):
    t = n*dt
    
    # Store old solution
    u_old = u.copy()
    
    # Source terms
    source = -np.pi*c*np.exp(-t)*np.cos(np.pi*x) + np.exp(-t)*np.sin(np.pi*x)
    
    # First order upwind scheme with source terms
    for i in range(1, nx-1):
        u[i] = u_old[i] - c*dt/dx*(u_old[i] - u_old[i-1]) + dt*source[i]
    
    # Apply boundary conditions
    u[0] = 0
    u[-1] = 0
    
    # Store solutions at specific time steps
    if abs(t - T/4) < dt/2:
        u_quarter = u.copy()
    elif abs(t - T/2) < dt/2:
        u_half = u.copy()
    elif abs(t - T) < dt/2:
        u_final = u.copy()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, u_initial, 'b-', label='t = 0')
plt.plot(x, u_quarter, 'g--', label=f't = {T/4:.1f}')
plt.plot(x, u_half, 'r-.', label=f't = {T/2:.1f}')
plt.plot(x, u_final, 'k:', label=f't = {T:.1f}')
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Linear Convection - First Order Upwind')
plt.legend()
plt.grid(True)
plt.show()

# Print CFL number
print(f"CFL number: {CFL}")
print(f"dx: {dx:.6f}")
print(f"dt: {dt:.6f}")