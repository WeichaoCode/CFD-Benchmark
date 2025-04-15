import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
L = 2.0   # length of domain
T = 2.0   # total time
dx = L / (nx-1)
dt = T / nt
nu = 0.07  # viscosity

# Grid points
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Initialize solution array
u = np.zeros((nt, nx))

# Initial condition
u[0, :] = np.sin(np.pi * x)

# Source term function
def source(x, t):
    return (-np.pi**2 * nu * np.exp(-t) * np.sin(np.pi*x) + 
            np.exp(-t) * np.sin(np.pi*x) - 
            np.pi * np.exp(-2*t) * np.sin(np.pi*x) * np.cos(np.pi*x))

# First Order Upwind scheme
def upwind_step(u_prev):
    u_new = np.zeros_like(u_prev)
    
    # Apply boundary conditions
    u_new[0] = 0
    u_new[-1] = 0
    
    # Update interior points
    for i in range(1, nx-1):
        # Convective term (upwind)
        if u_prev[i] >= 0:
            conv = u_prev[i] * (u_prev[i] - u_prev[i-1]) / dx
        else:
            conv = u_prev[i] * (u_prev[i+1] - u_prev[i]) / dx
        
        # Diffusive term
        diff = nu * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1]) / dx**2
        
        # Source term
        src = source(x[i], t[n])
        
        # Update solution
        u_new[i] = u_prev[i] + dt * (-conv + diff + src)
    
    return u_new

# Time stepping
for n in range(nt-1):
    u[n+1] = upwind_step(u[n])

# Plot results at specified time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0], 'b-', label='t = 0')
plt.plot(x, u[int(nt/4)], 'r--', label=f't = {T/4:.2f}')
plt.plot(x, u[int(nt/2)], 'g-.', label=f't = {T/2:.2f}')
plt.plot(x, u[-1], 'k:', label=f't = {T:.2f}')

plt.title("1D Burgers' Equation - First Order Upwind")
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Print stability condition
max_velocity = np.max(np.abs(u))
cfl = max_velocity * dt / dx
diff_number = nu * dt / dx**2

print(f"\nStability Analysis:")
print(f"CFL number: {cfl:.3f} (should be ≤ 1)")
print(f"Diffusion number: {diff_number:.3f} (should be ≤ 0.5)")

