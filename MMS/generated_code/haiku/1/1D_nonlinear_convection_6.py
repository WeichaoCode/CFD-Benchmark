import numpy as np
import matplotlib.pyplot as plt

def beam_warming_1d_nonlinear_convection():
    # Domain parameters
    x_start, x_end = 0, 2
    t_start, t_end = 0, 2
    
    # Mesh parameters
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
    
    # Beam-Warming method
    for n in range(nt - 1):
        for i in range(2, nx - 2):
            # Time-dependent source terms
            source_term = np.exp(-t[n]) * np.sin(np.pi * x[i]) - \
                          np.pi * np.exp(-2 * t[n]) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i])
            
            # Nonlinear convection term using Beam-Warming
            u_i = u[n, i]
            u_i_minus_1 = u[n, i-1]
            u_i_minus_2 = u[n, i-2]
            
            # Beam-Warming discretization
            convection_term = u_i * (3 * u_i - 4 * u_i_minus_1 + u_i_minus_2) / (2 * dx)
            
            # Update solution
            u[n+1, i] = u[n, i] - dt * convection_term + dt * source_term
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title('1D Nonlinear Convection - Beam-Warming Method')
    plt.xlabel('x')
    plt.ylabel('u')
    
    time_steps = [0, nt//4, nt//2, nt-1]
    for idx in time_steps:
        plt.plot(x, u[idx, :], label=f't = {t[idx]:.2f}')
    
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the simulation
beam_warming_1d_nonlinear_convection()