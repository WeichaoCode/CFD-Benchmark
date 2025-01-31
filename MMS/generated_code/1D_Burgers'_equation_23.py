import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0 / (nx - 1)
dt = 2.0 / (nt - 1)
nu = 0.07  # viscosity
x = np.linspace(0, 2, nx)
t = np.linspace(0, 2, nt)

# Stability check (CFL condition)
c = dt / dx
if c > 1:
    print("Warning: Solution might be unstable! CFL > 1")

# Initialize solution array
u = np.zeros((nt, nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Source term function
def source(x, t):
    return (-np.pi**2 * nu * np.exp(-t) * np.sin(np.pi * x) + 
            np.exp(-t) * np.sin(np.pi * x) - 
            np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x))

# Lax-Wendroff scheme
def lax_wendroff_step(u_prev, dt, dx, nu, t_current):
    u_new = np.zeros_like(u_prev)
    n = len(u_prev)
    
    # Interior points
    for i in range(1, n-1):
        # First order terms
        flux_diff = -(u_prev[i] * (u_prev[i+1] - u_prev[i-1])) / (2*dx)
        
        # Second order terms
        diffusion = nu * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1]) / (dx**2)
        
        # Source term
        src = source(x[i], t_current)
        
        u_new[i] = u_prev[i] + dt * (flux_diff + diffusion + src)
    
    # Boundary conditions
    u_new[0] = 0
    u_new[-1] = 0
    
    return u_new

# Time stepping
for n in range(0, nt-1):
    u[n+1] = lax_wendroff_step(u[n], dt, dx, nu, t[n])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, u[0], 'b-', label='t = 0')
plt.plot(x, u[nt//4], 'r--', label=f't = {t[nt//4]:.2f}')
plt.plot(x, u[nt//2], 'g-.', label=f't = {t[nt//2]:.2f}')
plt.plot(x, u[-1], 'k:', label=f't = {t[-1]:.2f}')

plt.title("1D Burgers' Equation - Lax-Wendroff Method")
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()

# Print max values at different times for verification
print(f"Max value at t=0: {np.max(np.abs(u[0])):.4f}")
print(f"Max value at t=T/4: {np.max(np.abs(u[nt//4])):.4f}")
print(f"Max value at t=T/2: {np.max(np.abs(u[nt//2])):.4f}")
print(f"Max value at t=T: {np.max(np.abs(u[-1])):.4f}")