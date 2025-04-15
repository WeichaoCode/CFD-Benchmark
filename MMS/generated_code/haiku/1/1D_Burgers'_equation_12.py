import numpy as np
import matplotlib.pyplot as plt

def exact_solution(x, t, nu):
    """Compute the exact solution of Burgers' equation."""
    return np.sin(np.pi * x) * np.exp(-t)

def beam_warming_burgers(nu, x_domain, t_domain, nx, nt):
    """
    Solve 1D Burgers' equation using Beam-Warming method
    
    Parameters:
    nu: viscosity
    x_domain: spatial domain [x_min, x_max]
    t_domain: temporal domain [t_min, t_max]
    nx: number of spatial points
    nt: number of temporal points
    """
    # Grid setup
    x_min, x_max = x_domain
    t_min, t_max = t_domain
    
    dx = (x_max - x_min) / (nx - 1)
    dt = (t_max - t_min) / (nt - 1)
    
    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(t_min, t_max, nt)
    
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
            # Nonlinear terms
            u_x_minus1 = (u[n, i] - u[n, i-1]) / dx
            u_x_minus2 = (u[n, i-1] - u[n, i-2]) / dx
            
            # Diffusion term
            u_xx = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / (dx**2)
            
            # Source terms
            source = np.pi * nu * np.exp(-t[n]) * np.sin(np.pi * x[i]) \
                     - np.exp(-t[n]) * np.sin(np.pi * x[i]) \
                     + np.pi * np.exp(-2*t[n]) * np.sin(np.pi * x[i]) * np.cos(np.pi * x[i])
            
            # Beam-Warming scheme
            u[n+1, i] = u[n, i] - dt * (
                u[n, i] * u_x_minus1 
                - nu * u_xx
                + source
            )
    
    return x, t, u

def plot_solution(x, t, u, equation_name):
    """Plot solution at key time steps"""
    key_times = [0, len(t)//4, len(t)//2, -1]
    plt.figure(figsize=(10, 6))
    
    for idx in key_times:
        plt.plot(x, u[idx, :], label=f't = {t[idx]:.2f}')
    
    plt.title(f'{equation_name} - Beam-Warming Method')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    plt.show()

# Simulation parameters
nu = 0.07
x_domain = [0, 2]
t_domain = [0, 2]
nx = 100
nt = 200

# Solve and plot
x, t, u = beam_warming_burgers(nu, x_domain, t_domain, nx, nt)
plot_solution(x, t, u, '1D Burgers\' Equation')