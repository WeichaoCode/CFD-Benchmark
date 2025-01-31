import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0 / (nx - 1)  # spatial step size
dt = 2.0 / nt  # time step size
nu = 0.07  # viscosity
x = np.linspace(0, 2, nx)  # spatial domain
t = np.linspace(0, 2, nt)  # temporal domain

# Initialize solution array
u = np.zeros((nt, nx))

# Set initial condition
u[0, :] = np.sin(np.pi * x)

# Beam-Warming scheme
def beam_warming_step(u_prev, dx, dt, nu):
    u_new = np.zeros_like(u_prev)
    
    # Interior points
    for i in range(2, nx-1):
        # Spatial derivatives
        u_x = (3*u_prev[i] - 4*u_prev[i-1] + u_prev[i-2])/(2*dx)
        u_xx = (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1])/dx**2
        
        # Source terms
        source = (-np.pi**2 * nu * np.exp(-t[n]) * np.sin(np.pi * x[i]) + 
                 np.exp(-t[n]) * np.sin(np.pi * x[i]) - 
                 np.pi * np.exp(-2*t[n]) * np.sin(np.pi * x[i]) * 
                 np.cos(np.pi * x[i]))
        
        # Update solution
        u_new[i] = (u_prev[i] - dt * u_prev[i] * u_x + 
                   dt * nu * u_xx + dt * source)
    
    # Boundary conditions
    u_new[0] = 0
    u_new[-1] = 0
    
    return u_new

# Time stepping
for n in range(nt-1):
    u[n+1] = beam_warming_step(u[n], dx, dt, nu)

# Plot results at key time steps
plt.figure(figsize=(10, 6))
key_times = [0, nt//4, nt//2, nt-1]
labels = ['t = 0', 't = T/4', 't = T/2', 't = T']

for i, n in enumerate(key_times):
    plt.plot(x, u[n], label=labels[i])

plt.title("1D Burgers' Equation - Beam-Warming Scheme")
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Stability analysis
# CFL condition: dt <= dx/max(|u|)
max_velocity = np.max(np.abs(u))
cfl = max_velocity * dt/dx
print(f"CFL number: {cfl:.3f}")

# von Neumann stability condition for viscous term
von_neumann = nu * dt/dx**2
print(f"von Neumann number: {von_neumann:.3f}")