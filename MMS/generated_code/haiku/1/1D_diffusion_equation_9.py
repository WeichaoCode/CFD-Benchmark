import numpy as np
import matplotlib.pyplot as plt
import math

def exact_solution(x, t, nu):
    """Compute the exact solution for comparison."""
    return np.exp(-t) * np.sin(np.pi * x)

def beam_warming_diffusion(nu, nx, nt, x_start, x_end, t_start, t_end):
    """
    Solve 1D diffusion equation using Beam-Warming method
    
    Parameters:
    - nu: viscosity
    - nx: number of spatial points
    - nt: number of temporal points
    - x_start, x_end: spatial domain
    - t_start, t_end: temporal domain
    """
    # Grid spacing
    dx = (x_end - x_start) / (nx - 1)
    dt = (t_end - t_start) / (nt - 1)
    
    # Stability check (von Neumann analysis)
    stability_condition = nu * dt / (dx**2)
    if stability_condition > 0.5:
        raise ValueError(f"Unstable scheme. Reduce time step. Current condition: {stability_condition}")
    
    # Initialize grid
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
    for n in range(1, nt):
        for i in range(1, nx-1):
            # Source term
            source = np.pi * nu * np.exp(-t[n-1]) * np.sin(np.pi * x[i]) - np.exp(-t[n-1]) * np.sin(np.pi * x[i])
            
            # Beam-Warming discretization
            u[n, i] = u[n-1, i] + dt * (
                nu * (u[n-1, i+1] - 2*u[n-1, i] + u[n-1, i-1]) / (dx**2) 
                + source
            )
    
    return x, t, u

def main():
    # Problem parameters
    nu = 0.3
    nx = 50  # spatial points
    nt = 100  # temporal points
    x_start, x_end = 0, 2
    t_start, t_end = 0, 2
    
    # Solve PDE
    x, t, u = beam_warming_diffusion(nu, nx, nt, x_start, x_end, t_start, t_end)
    
    # Plot solution at key time steps
    plt.figure(figsize=(10, 6))
    plt.title('1D Diffusion Equation - Beam-Warming Method')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    
    # Time steps to plot
    time_indices = [0, int(nt/4), int(nt/2), -1]
    labels = [f't = {t[idx]:.2f}' for idx in time_indices]
    
    for idx, label in zip(time_indices, labels):
        plt.plot(x, u[idx, :], label=label)
    
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()