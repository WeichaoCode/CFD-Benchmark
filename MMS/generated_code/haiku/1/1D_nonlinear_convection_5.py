import numpy as np
import matplotlib.pyplot as plt

def initial_condition(x):
    return np.sin(np.pi * x)

def boundary_conditions(u, t):
    u[0] = 0  # Left boundary
    u[-1] = 0  # Right boundary
    return u

def source_term(x, t):
    return np.exp(-t) * np.sin(np.pi * x) - np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x)

def lax_friedrichs_solver(nx, nt, dx, dt, T):
    # Initialize grid
    x = np.linspace(0, 2, nx)
    t = np.linspace(0, T, nt)
    
    # Initialize solution array
    u = np.zeros((nt, nx))
    
    # Initial condition
    u[0, :] = initial_condition(x)
    
    # Time stepping
    for n in range(nt-1):
        # Copy previous time step
        u_old = u[n].copy()
        
        # Lax-Friedrichs method
        for j in range(1, nx-1):
            u[n+1, j] = 0.5 * (u_old[j-1] + u_old[j+1]) - \
                        0.5 * (dt/dx) * (u_old[j+1]**2/2 - u_old[j-1]**2/2) + \
                        dt * source_term(x[j], t[n])
        
        # Apply boundary conditions
        u[n+1, :] = boundary_conditions(u[n+1, :], t[n+1])
    
    return x, t, u

def von_neumann_stability_analysis(dx, dt):
    # CFL condition for Lax-Friedrichs method
    max_velocity = 1.0  # Assuming max characteristic speed
    cfl = max_velocity * dt / dx
    
    print(f"CFL Number: {cfl}")
    if cfl > 1:
        print("Warning: Scheme might be unstable!")
    else:
        print("Scheme is stable.")
    
    return cfl <= 1

def plot_solution(x, t, u):
    plt.figure(figsize=(10, 6))
    time_indices = [0, int(len(t)/4), int(len(t)/2), -1]
    
    for idx in time_indices:
        plt.plot(x, u[idx], label=f't = {t[idx]:.2f}')
    
    plt.title('1D Nonlinear Convection - Lax-Friedrichs Method')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Simulation parameters
    nx = 100  # Spatial grid points
    nt = 200  # Time steps
    T = 2.0   # Total simulation time
    
    # Grid spacing
    dx = 2 / (nx - 1)
    dt = T / (nt - 1)
    
    # Stability check
    von_neumann_stability_analysis(dx, dt)
    
    # Solve PDE
    x, t, u = lax_friedrichs_solver(nx, nt, dx, dt, T)
    
    # Plot solution
    plot_solution(x, t, u)

if __name__ == "__main__":
    main()