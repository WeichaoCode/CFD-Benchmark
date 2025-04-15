import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0/(nx-1)  # spatial step size
dt = 2.0/nt  # time step size
c = 1.0  # wave speed

# Check CFL condition for stability
CFL = c*dt/dx
print(f"CFL number: {CFL}")
if CFL > 1:
    print("Warning: Solution might be unstable! CFL > 1")

# Initialize arrays
x = np.linspace(0, 2, nx)
t = np.linspace(0, 2, nt)
u = np.zeros((nt, nx))

# Initial condition
u[0,:] = np.sin(np.pi*x)

# Source term function
def source(x, t):
    return -np.pi*c*np.exp(-t)*np.cos(np.pi*x) + np.exp(-t)*np.sin(np.pi*x)

# First Order Upwind scheme
def upwind_step(u_prev, dx, dt, c):
    u_new = np.zeros_like(u_prev)
    # Apply boundary conditions
    u_new[0] = 0
    u_new[-1] = 0
    
    # Update interior points
    for i in range(1, len(u_prev)-1):
        u_new[i] = u_prev[i] - c*dt/dx*(u_prev[i] - u_prev[i-1]) + dt*source(x[i], t[j])
    
    return u_new

# Time stepping
for j in range(nt-1):
    u[j+1,:] = upwind_step(u[j,:], dx, dt, c)

# Plot results at specific time steps
plt.figure(figsize=(10, 6))
t_plot = [0, int(nt/4), int(nt/2), nt-1]
labels = ['t = 0', 't = T/4', 't = T/2', 't = T']
colors = ['b', 'r', 'g', 'k']

for i, t_idx in enumerate(t_plot):
    plt.plot(x, u[t_idx,:], colors[i], label=labels[i])

plt.title('1D Linear Convection - First Order Upwind')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Calculate and print maximum error at final time
exact_solution = np.exp(-2)*np.sin(np.pi*x)  # analytical solution at t = T
error = np.max(np.abs(u[-1,:] - exact_solution))
print(f"Maximum error at t = T: {error}")