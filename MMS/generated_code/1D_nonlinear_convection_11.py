import numpy as np
import matplotlib.pyplot as plt

def source_term(x, t):
    """Source term in the equation"""
    return np.exp(-t) * np.sin(np.pi*x) - np.pi * np.exp(-2*t) * np.sin(np.pi*x) * np.cos(np.pi*x)

def lax_wendroff_step(u, dx, dt, t, x):
    """Perform one step of Lax-Wendroff method"""
    n = len(u)
    u_new = np.zeros(n)
    
    # Interior points
    for i in range(1, n-1):
        # First term (time derivative)
        first_term = -0.5 * dt/(2*dx) * (u[i+1]**2 - u[i-1]**2)
        
        # Second term (spatial derivative squared)
        second_term = 0.5 * (dt/dx)**2 * (
            u[i]*(u[i+1]**2 - u[i-1]**2)/4
        )
        
        # Source term
        s = source_term(x[i], t)
        
        u_new[i] = u[i] + first_term + second_term + dt*s
    
    # Boundary conditions
    u_new[0] = 0
    u_new[-1] = 0
    
    return u_new

# Parameters
nx = 100  # Number of spatial points
nt = 1000  # Number of time steps
x_min, x_max = 0, 2
t_min, t_max = 0, 2

# Grid
dx = (x_max - x_min)/(nx-1)
dt = (t_max - t_min)/nt
x = np.linspace(x_min, x_max, nx)
t = np.linspace(t_min, t_max, nt)

# Initial condition
u = np.sin(np.pi*x)

# CFL condition check
max_velocity = max(abs(u))
cfl = max_velocity * dt/dx
print(f"CFL number: {cfl}")
if cfl > 1:
    print("Warning: CFL condition not satisfied!")

# Storage for solution at key time steps
key_times = [0, t_max/4, t_max/2, t_max]
solutions = {0: u.copy()}

# Time stepping
current_time = 0
for n in range(nt):
    current_time += dt
    u = lax_wendroff_step(u, dx, dt, current_time, x)
    
    # Store solution at key times
    for key_time in key_times:
        if abs(current_time - key_time) < dt/2:
            solutions[key_time] = u.copy()

# Plotting
plt.figure(figsize=(10, 6))
for time in key_times:
    if time in solutions:
        plt.plot(x, solutions[time], label=f't = {time:.2f}')

plt.title('1D Nonlinear Convection - Lax-Wendroff Method')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()