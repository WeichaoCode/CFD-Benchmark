import numpy as np
import matplotlib.pyplot as plt

def solve_1d_convection():
    # Parameters
    L = 2.0  # Length of domain
    T = 2.0  # Total time
    c = 1.0  # Wave speed
    
    # Numerical parameters
    nx = 100  # Number of spatial points
    nt = 200  # Number of time points
    dx = L / (nx-1)  # Spatial step size
    dt = T / (nt-1)  # Time step size
    
    # Grid points
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    
    # Initialize solution array
    u = np.zeros((nt, nx))
    
    # Set initial condition
    u[0, :] = np.sin(np.pi * x)
    
    # Set boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0
    
    # CFL number
    CFL = c * dt / dx
    
    # Time stepping
    for n in range(0, nt-1):
        for i in range(1, nx-1):
            # Source term
            source = -np.pi * c * np.exp(-t[n]) * np.cos(np.pi * x[i]) + np.exp(-t[n]) * np.sin(np.pi * x[i])
            
            # FTCS scheme
            u[n+1, i] = u[n, i] - 0.5 * CFL * (u[n, i+1] - u[n, i-1]) + dt * source
    
    return x, t, u

def plot_results(x, t, u):
    # Plot initial condition
    plt.figure(figsize=(10, 6))
    plt.plot(x, u[0, :], 'b-', label='t = 0')
    
    # Plot solution at different times
    t_points = [50, 100, 150, -1]
    colors = ['r-', 'g-', 'm-', 'k-']
    labels = ['t = 0.5', 't = 1.0', 't = 1.5', 't = 2.0']
    
    for i, t_point in enumerate(t_points):
        plt.plot(x, u[t_point, :], colors[i], label=labels[i])
    
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('1D Linear Convection')
    plt.legend()
    plt.grid(True)
    plt.show()

# Solve and plot
x, t, u = solve_1d_convection()
plot_results(x, t, u)

# Print maximum and minimum values
print(f"Maximum value: {np.max(u)}")
print(f"Minimum value: {np.min(u)}")