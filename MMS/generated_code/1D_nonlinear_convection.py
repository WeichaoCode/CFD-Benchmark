import numpy as np
import matplotlib.pyplot as plt

def solve_1d_nonlinear_convection():
    # Parameters
    Lx = 2.0  # Length of spatial domain
    T = 2.0   # Total time
    nx = 100  # Number of spatial points
    nt = 1000 # Number of time steps
    
    # Grid parameters
    dx = Lx / (nx-1)
    dt = T / (nt-1)
    
    # Create grid
    x = np.linspace(0, Lx, nx)
    t = np.linspace(0, T, nt)
    
    # Initialize solution array
    u = np.zeros((nt, nx))
    
    # Set initial condition
    u[0, :] = np.sin(np.pi * x)
    
    # Set boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0
    
    # Time stepping
    for n in range(0, nt-1):
        for i in range(1, nx-1):
            # Source terms
            source = np.exp(-t[n]) * np.sin(np.pi * x[i]) - \
                    np.pi * np.exp(-2*t[n]) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i])
            
            # Central difference for spatial derivative
            dudx = (u[n, i+1] - u[n, i-1]) / (2*dx)
            
            # Update solution
            u[n+1, i] = u[n, i] - dt * (u[n, i] * dudx - source)
    
    return x, t, u

def plot_results(x, t, u):
    # Plot results at different time steps
    plt.figure(figsize=(10, 6))
    
    # Plot at different times
    time_points = [0, int(len(t)/4), int(len(t)/2), len(t)-1]
    labels = ['t=0', 't=0.5', 't=1.0', 't=2.0']
    
    for i, n in enumerate(time_points):
        plt.plot(x, u[n, :], label=labels[i])
    
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('1D Nonlinear Convection')
    plt.legend()
    plt.grid(True)
    plt.show()

# Solve and plot
x, t, u = solve_1d_nonlinear_convection()
plot_results(x, t, u)

# Print max and min values
print(f"Maximum value: {np.max(u)}")
print(f"Minimum value: {np.min(u)}")