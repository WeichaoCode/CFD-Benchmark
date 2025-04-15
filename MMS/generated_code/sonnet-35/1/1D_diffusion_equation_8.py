import numpy as np
import matplotlib.pyplot as plt

def solve_diffusion_equation():
    # Parameters
    L = 2.0  # Length of domain
    T = 2.0  # Total time
    nu = 0.3  # Viscosity
    
    # Discretization
    Nx = 100  # Number of spatial points
    dx = L / (Nx-1)
    
    # Calculate dt based on stability condition (von Neumann analysis)
    # For Lax-Friedrichs: dt ≤ dx²/(2*nu)
    dt = 0.8 * dx**2 / (2*nu)  # Using safety factor of 0.8
    Nt = int(T/dt)
    
    # Create grid
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    
    # Initialize solution array
    u = np.zeros((Nt, Nx))
    
    # Set initial condition
    u[0,:] = np.sin(np.pi*x)
    
    # Time stepping using Lax-Friedrichs method
    for n in range(0, Nt-1):
        for i in range(1, Nx-1):
            # Source term
            source = -np.pi**2 * nu * np.exp(-t[n]) * np.sin(np.pi*x[i]) + \
                    np.exp(-t[n]) * np.sin(np.pi*x[i])
            
            # Lax-Friedrichs scheme
            u[n+1,i] = 0.5*(u[n,i+1] + u[n,i-1]) + \
                       nu*dt/(dx**2) * (u[n,i+1] - 2*u[n,i] + u[n,i-1]) + \
                       dt*source
        
        # Apply boundary conditions
        u[n+1,0] = 0
        u[n+1,-1] = 0
    
    return x, t, u

def plot_solution(x, t, u):
    plt.figure(figsize=(10, 6))
    
    # Plot solution at different time steps
    plt.plot(x, u[0,:], 'b-', label='t = 0')
    plt.plot(x, u[int(len(t)/4),:], 'r--', label=f't = {t[int(len(t)/4)]:.2f}')
    plt.plot(x, u[int(len(t)/2),:], 'g-.', label=f't = {t[int(len(t)/2)]:.2f}')
    plt.plot(x, u[-1,:], 'k:', label=f't = {t[-1]:.2f}')
    
    plt.title('1D Diffusion Equation - Lax-Friedrichs Method')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Solve and plot
x, t, u = solve_diffusion_equation()
plot_solution(x, t, u)

# Print stability parameters
dx = x[1] - x[0]
dt = t[1] - t[0]
print(f"\nNumerical Parameters:")
print(f"dx = {dx:.6f}")
print(f"dt = {dt:.6f}")
print(f"Stability parameter (dt/(dx^2)) = {dt/dx**2:.6f}")