import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of domain
T = 2.0  # Total time
nu = 0.07  # Viscosity
nx = 100  # Number of spatial points
nt = 1000  # Number of time steps
dx = L / (nx - 1)
dt = T / nt

# Grid points
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Initialize solution array
u = np.zeros((nt, nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Source term function
def source(x, t):
    return (-np.pi**2 * nu * np.exp(-t) * np.sin(np.pi * x) + 
            np.exp(-t) * np.sin(np.pi * x) - 
            np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x))

# First Order Upwind scheme
def upwind_step(u_prev, dx, dt, nu):
    u_new = np.zeros_like(u_prev)
    u_new[0] = 0  # Boundary condition at x = 0
    u_new[-1] = 0  # Boundary condition at x = 2
    
    for i in range(1, len(u_prev)-1):
        # Convective term
        if u_prev[i] >= 0:
            conv = u_prev[i] * (u_prev[i] - u_prev[i-1]) / dx
        else:
            conv = u_prev[i] * (u_prev[i+1] - u_prev[i]) / dx
        
        # Diffusive term
        diff = nu * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1]) / dx**2
        
        # Source term
        src = source(x[i], t[j])
        
        # Update solution
        u_new[i] = u_prev[i] + dt * (-conv + diff + src)
    
    return u_new

# Time stepping
for j in range(nt-1):
    u[j+1] = upwind_step(u[j], dx, dt, nu)

# Plot results at key time steps
plt.figure(figsize=(10, 6))
key_times = [0, int(nt/4), int(nt/2), nt-1]
labels = ['t = 0', 't = T/4', 't = T/2', 't = T']
for i, kt in enumerate(key_times):
    plt.plot(x, u[kt], label=labels[i])

plt.title("1D Burgers' Equation - First Order Upwind Scheme")
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Print maximum CFL number for stability check
max_velocity = np.max(np.abs(u))
CFL = max_velocity * dt / dx
print(f"Maximum CFL number: {CFL:.3f}")
print(f"For stability, CFL should be <= 1")

# Print diffusion number for stability check
D = nu * dt / dx**2
print(f"Diffusion number: {D:.3f}")
print(f"For stability, diffusion number should be <= 0.5")