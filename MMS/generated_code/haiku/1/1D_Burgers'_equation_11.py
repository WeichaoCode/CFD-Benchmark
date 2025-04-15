import numpy as np
import matplotlib.pyplot as plt

def burgers_lax_friedrichs():
    # Physical parameters
    nu = 0.07  # Viscosity
    L = 2.0    # Spatial domain length
    T = 2.0    # Total simulation time

    # Numerical parameters
    nx = 100   # Number of spatial points
    nt = 200   # Number of time steps

    # Grid setup
    dx = L / (nx - 1)
    dt = T / (nt - 1)
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)

    # Initialize solution array
    u = np.zeros((nt, nx))

    # Initial condition
    u[0, :] = np.sin(np.pi * x)

    # Boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0

    # Source term function
    def source_term(x, t):
        return (np.pi * nu * np.exp(-t) * np.sin(np.pi * x) +
                np.exp(-t) * np.sin(np.pi * x) -
                np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x))

    # Lax-Friedrichs method
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            # Lax-Friedrichs flux
            u_left = u[n, i-1]
            u_right = u[n, i+1]
            
            # Viscous term
            viscous_term = nu * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / (dx**2)
            
            # Convective term
            convective_term = 0.5 * (u[n, i+1]**2 - u[n, i-1]**2) / (2*dx)
            
            # Source term
            src = source_term(x[i], t[n])
            
            # Update solution
            u[n+1, i] = u[n, i] - dt * (convective_term + viscous_term) + dt * src

    # Plotting
    plt.figure(figsize=(10, 6))
    time_steps = [0, nt//4, nt//2, -1]
    for idx, ts in enumerate(time_steps):
        plt.plot(x, u[ts, :], label=f't = {t[ts]:.2f}')

    plt.title('1D Burgers\' Equation - Lax-Friedrichs Method')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    plt.show()

    return u

# Run simulation
solution = burgers_lax_friedrichs()