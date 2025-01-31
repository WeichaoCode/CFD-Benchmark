import numpy as np
import matplotlib.pyplot as plt

def beam_warming_scheme(nx, nt, dx, dt, x, t):
    # Initialize solution array
    u = np.zeros((nt, nx))
    
    # Set initial condition
    u[0, :] = np.sin(np.pi * x)
    
    # Time stepping
    for n in range(0, nt-1):
        for i in range(2, nx):
            # Source term
            source = np.exp(-t[n]) * np.sin(np.pi * x[i]) - \
                    np.pi * np.exp(-2*t[n]) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i])
            
            # Beam-Warming scheme for nonlinear convection
            if u[n, i] >= 0:
                u[n+1, i] = u[n, i] - \
                           (dt/dx) * u[n, i] * (3*u[n, i] - 4*u[n, i-1] + u[n, i-2])/2 + \
                           dt * source
            else:
                u[n+1, i] = u[n, i] - \
                           (dt/dx) * u[n, i] * (-u[n, i+2] + 4*u[n, i+1] - 3*u[n, i])/2 + \
                           dt * source
        
        # Apply boundary conditions
        u[n+1, 0] = 0
        u[n+1, -1] = 0
        
    return u

# Set up grid parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
nx = 100  # Number of spatial points
nt = 200  # Number of time points

dx = L/(nx-1)
dt = T/(nt-1)

# Create spatial and temporal grids
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Check CFL condition for stability
max_u = 1.0  # Maximum expected velocity
CFL = max_u * dt/dx
print(f"CFL number: {CFL}")

# Solve PDE
u = beam_warming_scheme(nx, nt, dx, dt, x, t)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], 'b-', label='t = 0')
plt.plot(x, u[nt//4, :], 'g-', label=f't = {T/4:.1f}')
plt.plot(x, u[nt//2, :], 'r-', label=f't = {T/2:.1f}')
plt.plot(x, u[-1, :], 'k-', label=f't = {T:.1f}')

plt.title('1D Nonlinear Convection - Beam-Warming Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()

# Print stability information
print("\nStability Analysis:")
print(f"dx = {dx:.6f}")
print(f"dt = {dt:.6f}")
print("The Beam-Warming scheme is stable for CFL ≤ 1")
if CFL <= 1:
    print("The current scheme is stable")
else:
    print("Warning: The current scheme might be unstable")