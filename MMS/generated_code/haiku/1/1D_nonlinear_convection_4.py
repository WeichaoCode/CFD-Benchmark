import numpy as np
import matplotlib.pyplot as plt

def nonlinear_convection_1d():
    # Domain parameters
    x_start, x_end = 0, 2
    t_start, t_end = 0, 2
    
    # Numerical parameters
    nx = 100  # spatial points
    nt = 200  # time steps
    
    # Grid generation
    dx = (x_end - x_start) / (nx - 1)
    dt = (t_end - t_start) / (nt - 1)
    
    x = np.linspace(x_start, x_end, nx)
    t = np.linspace(t_start, t_end, nt)
    
    # Initialize solution array
    u = np.zeros((nt, nx))
    
    # Initial condition
    u[0, :] = np.sin(np.pi * x)
    
    # Boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0
    
    # Stability check (von Neumann analysis)
    if dt/dx > 1:
        raise ValueError("Unstable scheme: Courant number > 1")
    
    # Numerical solution using First Order Upwind
    for n in range(nt-1):
        for i in range(1, nx-1):
            u_n = u[n, i]
            t_n = t[n]
            
            # Nonlinear convection term
            if u_n >= 0:
                du_dx_upwind = (u_n - u[n, i-1]) / dx
            else:
                du_dx_upwind = (u[n, i+1] - u_n) / dx
            
            # Additional source terms
            source_term = np.exp(-t_n) * np.sin(np.pi * x[i]) - \
                          np.pi * np.exp(-2*t_n) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i])
            
            # Update solution
            u[n+1, i] = u_n - dt * (u_n * du_dx_upwind + source_term)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title('1D Nonlinear Convection - First Order Upwind')
    
    time_indices = [0, nt//4, nt//2, nt-1]
    labels = [f't = {t[idx]:.2f}' for idx in time_indices]
    
    for idx in time_indices:
        plt.plot(x, u[idx, :], label=labels[idx])
    
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run simulation
nonlinear_convection_1d()