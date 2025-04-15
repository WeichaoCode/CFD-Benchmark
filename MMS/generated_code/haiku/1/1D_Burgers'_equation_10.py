import numpy as np
import matplotlib.pyplot as plt

def burgers_equation_fou_solver():
    # Physical parameters
    nu = 0.07  # viscosity
    L = 2.0    # spatial domain length
    T = 2.0    # temporal domain length

    # Numerical parameters
    nx = 100   # spatial grid points
    nt = 200   # temporal grid points
    dx = L / (nx - 1)
    dt = T / (nt - 1)

    # Stability condition (CFL)
    CFL = 0.5
    
    # Grid initialization
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    
    # Initial condition
    u = np.sin(np.pi * x)
    
    # Solution storage
    solution = np.zeros((nt, nx))
    solution[0, :] = u
    
    # Source term function
    def source_term(x, t):
        return (np.pi * nu * np.exp(-t) * np.sin(np.pi * x) + 
                np.exp(-t) * np.sin(np.pi * x) - 
                np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x))
    
    # FOU solver
    for n in range(1, nt):
        u_old = u.copy()
        
        for i in range(1, nx-1):
            # Upwind flux
            u_flux_plus = max(u_old[i], 0) * u_old[i] - min(u_old[i], 0) * u_old[i+1]
            u_flux_minus = max(u_old[i-1], 0) * u_old[i-1] - min(u_old[i-1], 0) * u_old[i]
            
            # Diffusion term
            u_diff = nu * (u_old[i+1] - 2*u_old[i] + u_old[i-1]) / (dx**2)
            
            # Update with source term
            u[i] = u_old[i] - dt * (u_flux_plus - u_flux_minus) / dx + dt * u_diff + dt * source_term(x[i], t[n-1])
        
        # Boundary conditions
        u[0] = 0
        u[-1] = 0
        
        solution[n, :] = u
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title('1D Burgers\' Equation - First Order Upwind')
    
    time_indices = [0, nt//4, nt//2, -1]
    for idx in time_indices:
        plt.plot(x, solution[idx, :], label=f't = {t[idx]:.2f}')
    
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the solver
burgers_equation_fou_solver()