import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # Number of spatial points
nt = 1000  # Number of time steps
dx = 2.0 / (nx - 1)  # Spatial step size
dt = 2.0 / (nt - 1)  # Time step size
nu = 0.07  # Viscosity
x = np.linspace(0, 2, nx)  # Spatial domain
t = np.linspace(0, 2, nt)  # Temporal domain

# Initialize solution array
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Beam-Warming scheme
def beam_warming_step(u_prev):
    u_new = np.zeros_like(u_prev)
    
    # Apply boundary conditions
    u_new[0] = 0
    u_new[-1] = 0
    
    # Interior points
    for i in range(2, nx-1):
        # Source terms
        source = (-np.pi**2 * nu * np.exp(-t[n]) * np.sin(np.pi * x[i]) + 
                 np.exp(-t[n]) * np.sin(np.pi * x[i]) - 
                 np.pi * np.exp(-2*t[n]) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i]))
        
        # Beam-Warming discretization
        if u_prev[i] >= 0:
            convective = u_prev[i] * (3*u_prev[i] - 4*u_prev[i-1] + u_prev[i-2])/(2*dx)
        else:
            convective = u_prev[i] * (-u_prev[i+2] + 4*u_prev[i+1] - 3*u_prev[i])/(2*dx)
        
        diffusive = nu * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1])/(dx**2)
        
        u_new[i] = u_prev[i] + dt * (-convective + diffusive + source)
    
    return u_new

# Time stepping
for n in range(nt-1):
    u[n+1] = beam_warming_step(u[n])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, u[0], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4)], 'r--', label=f't = {t[int(nt/4)]:.2f}')
plt.plot(x, u[int(nt/2)], 'g-.', label=f't = {t[int(nt/2)]:.2f}')
plt.plot(x, u[-1], 'k:', label=f't = {t[-1]:.2f}')
plt.xlabel('x')
plt.ylabel('u')
plt.title("1D Burgers' Equation - Beam-Warming Scheme")
plt.legend()
plt.grid(True)
plt.show()

# Stability condition (CFL)
c = np.max(np.abs(u)) * dt/dx
d = nu * dt/dx**2
print(f"CFL number: {c:.3f}")
print(f"Diffusion number: {d:.3f}")