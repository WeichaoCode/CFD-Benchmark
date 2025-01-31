import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.3  # viscosity
L = 2.0   # length of domain
T = 2.0   # total time
Nx = 100  # number of spatial points
Nt = 1000 # number of time steps
dx = L / (Nx-1)
dt = T / Nt

# Grid points
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Initialize solution matrix
u = np.zeros((Nt, Nx))

# Set initial condition
u[0,:] = np.sin(np.pi * x)

# Set boundary conditions
u[:,0] = 0
u[:,-1] = 0

# Stability criterion for explicit scheme
r = nu * dt / (dx**2)
if r > 0.5:
    print(f"Warning: Scheme might be unstable! r = {r}")

# Time stepping using explicit scheme
for n in range(0, Nt-1):
    for i in range(1, Nx-1):
        # Finite difference approximation
        d2u_dx2 = (u[n,i+1] - 2*u[n,i] + u[n,i-1]) / dx**2
        source = -np.pi**2 * nu * np.exp(-t[n]) * np.sin(np.pi * x[i]) + \
                np.exp(-t[n]) * np.sin(np.pi * x[i])
        
        u[n+1,i] = u[n,i] + dt * (nu * d2u_dx2 + source)

# Plot results
def plot_solution(x, t, u):
    plt.figure(figsize=(10, 6))
    
    # Plot at different time steps
    time_steps = [0, int(Nt/4), int(Nt/2), Nt-1]
    labels = ['t=0', 't=T/4', 't=T/2', 't=T']
    
    for i, n in enumerate(time_steps):
        plt.plot(x, u[n,:], label=labels[i])
    
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('1D Diffusion Equation Solution')
    plt.grid(True)
    plt.legend()
    plt.show()

# Compute exact solution for comparison
def exact_solution(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

# Calculate error
def compute_error():
    max_error = 0
    for n in range(Nt):
        u_exact = exact_solution(x, t[n])
        error = np.max(np.abs(u[n,:] - u_exact))
        max_error = max(max_error, error)
    return max_error

# Plot solution
plot_solution(x, t, u)

# Print maximum error
max_error = compute_error()
print(f"Maximum absolute error: {max_error}")

# Plot error over time at x = L/2
mid_point = Nx//2
error_at_mid = np.abs(u[:,mid_point] - exact_solution(x[mid_point], t))
plt.figure(figsize=(10, 6))
plt.plot(t, error_at_mid)
plt.xlabel('Time')
plt.ylabel('Absolute Error')
plt.title('Error at x = L/2 over time')
plt.grid(True)
plt.show()