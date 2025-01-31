import numpy as np
import matplotlib.pyplot as plt

def solve_nonlinear_convection():
    # Parameters
    L = 2.0  # Length of domain
    T = 2.0  # Total time
    nx = 100  # Number of spatial points
    dx = L / (nx - 1)  # Spatial step size
    
    # CFL condition for stability (from von Neumann analysis)
    CFL = 0.5
    dt = CFL * dx  # Time step size
    nt = int(T / dt)  # Number of time steps
    
    # Grid points
    x = np.linspace(0, L, nx)
    
    # Initialize solution arrays
    u = np.zeros((nt + 1, nx))
    
    # Initial condition
    u[0, :] = np.sin(np.pi * x)
    
    # First time step using Forward Euler (needed for Leapfrog)
    u[1, :] = u[0, :]
    for i in range(1, nx-1):
        source = np.exp(-dt) * np.sin(np.pi * x[i]) - np.pi * np.exp(-2*dt) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i])
        u[1, i] = u[0, i] - dt * u[0, i] * (u[0, i+1] - u[0, i-1])/(2*dx) + dt * source
    
    # Boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0
    
    # Leapfrog scheme
    for n in range(1, nt):
        for i in range(1, nx-1):
            source = np.exp(-n*dt) * np.sin(np.pi * x[i]) - np.pi * np.exp(-2*n*dt) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i])
            u[n+1, i] = u[n-1, i] - 2*dt * u[n, i] * (u[n, i+1] - u[n, i-1])/(2*dx) + 2*dt * source
            
        # Apply boundary conditions
        u[n+1, 0] = 0
        u[n+1, -1] = 0
        
        # Apply Robert-Asselin filter to control computational mode
        alpha = 0.1
        for i in range(1, nx-1):
            u[n, i] = u[n, i] + alpha * (u[n+1, i] - 2*u[n, i] + u[n-1, i])
    
    return x, u, nt, dt

def plot_results(x, u, nt, dt):
    plt.figure(figsize=(10, 6))
    
    # Plot at different time steps
    t_indices = [0, int(nt/4), int(nt/2), nt]
    times = [0, 0.5, 1.0, 2.0]
    
    for idx, t_idx in enumerate(t_indices):
        plt.plot(x, u[t_idx, :], label=f't = {times[idx]}')
    
    plt.title('1D Nonlinear Convection - Leapfrog Method')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid(True)
    plt.legend()
    plt.show()

# Run simulation
x, u, nt, dt = solve_nonlinear_convection()
plot_results(x, u, nt, dt)