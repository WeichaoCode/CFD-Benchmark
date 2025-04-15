import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
nx = 100  # Number of spatial points
nt = 1000  # Number of time steps
dx = L / (nx - 1)  # Spatial step
dt = T / nt  # Time step
nu = 0.07  # Viscosity

# Grid points
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Initialize solution array
u = np.zeros((nt, nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Stability check (CFL condition)
c = dt / dx
if c > 1:
    print("Warning: CFL condition not satisfied!")

def source_term(x, t):
    """Source terms in the equation"""
    return (-np.pi**2 * nu * np.exp(-t) * np.sin(np.pi * x) + 
            np.exp(-t) * np.sin(np.pi * x) - 
            np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x))

# Lax-Friedrichs scheme
def lax_friedrichs_step(u_prev, dt, dx, nu):
    u_next = np.zeros_like(u_prev)
    
    # Apply scheme for interior points
    for i in range(1, nx-1):
        # Convective term
        conv = -0.25 * (u_prev[i+1]**2 - u_prev[i-1]**2) / dx
        
        # Diffusive term
        diff = nu * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1]) / dx**2
        
        # Source term
        source = source_term(x[i], t[j])
        
        # Lax-Friedrichs scheme
        u_next[i] = 0.5 * (u_prev[i+1] + u_prev[i-1]) + dt * (diff + conv + source)
    
    # Boundary conditions
    u_next[0] = 0
    u_next[-1] = 0
    
    return u_next

# Time stepping
for j in range(nt-1):
    u[j+1] = lax_friedrichs_step(u[j], dt, dx, nu)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, u[0], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4)], 'r--', label=f't = {T/4:.1f}')
plt.plot(x, u[int(nt/2)], 'g-.', label=f't = {T/2:.1f}')
plt.plot(x, u[-1], 'k:', label=f't = {T:.1f}')

plt.title("1D Burgers' Equation - Lax-Friedrichs Method")
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()