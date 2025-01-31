import numpy as np
import matplotlib.pyplot as plt

def solve_1d_nonlinear_convection():
    # Domain parameters
    L = 2.0  # Length of domain
    T = 2.0  # Total time
    nx = 100  # Number of spatial points
    dx = L / (nx - 1)
    
    # CFL condition for stability (from von Neumann analysis)
    CFL = 0.8
    dt = CFL * dx  # Time step
    nt = int(T / dt)  # Number of time steps
    
    # Grid points
    x = np.linspace(0, L, nx)
    
    # Initialize solution array
    u = np.zeros((nt + 1, nx))
    
    # Initial condition
    u[0, :] = np.sin(np.pi * x)
    
    # Boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0
    
    # Time stepping
    for n in range(nt):
        t = n * dt
        un = u[n, :].copy()
        
        # First order upwind scheme
        for i in range(1, nx-1):
            # Source terms
            source = np.exp(-t) * np.sin(np.pi * x[i]) - \
                    np.pi * np.exp(-2*t) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i])
            
            # Upwind scheme based on sign of velocity
            if un[i] >= 0:
                u[n+1, i] = un[i] - dt * un[i] * ((un[i] - un[i-1])/dx) + dt * source
            else:
                u[n+1, i] = un[i] - dt * un[i] * ((un[i+1] - un[i])/dx) + dt * source
                
        # Apply boundary conditions
        u[n+1, 0] = 0
        u[n+1, -1] = 0
    
    return x, u, dt, nt

def plot_results(x, u, nt):
    plt.figure(figsize=(10, 6))
    
    # Plot at different time steps
    plt.plot(x, u[0, :], 'b-', label='t = 0')
    plt.plot(x, u[nt//4, :], 'r--', label=f't = T/4')
    plt.plot(x, u[nt//2, :], 'g-.', label=f't = T/2')
    plt.plot(x, u[-1, :], 'k:', label=f't = T')
    
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('1D Nonlinear Convection - First Order Upwind')
    plt.legend()
    plt.grid(True)
    plt.show()

# Solve and plot
x, u, dt, nt = solve_1d_nonlinear_convection()
plot_results(x, u, nt)