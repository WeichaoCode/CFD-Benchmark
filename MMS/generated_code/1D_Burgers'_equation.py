import numpy as np

def solve_burgers_1d():
    # Parameters
    nu = 0.07  # viscosity
    L = 2.0    # domain length
    T = 2.0    # total time
    
    # Discretization
    Nx = 100   # number of spatial points
    Nt = 1000  # number of time steps
    dx = L / (Nx - 1)
    dt = T / Nt
    
    # Grid points
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    
    # Initialize solution array
    u = np.zeros((Nt, Nx))
    
    # Set initial condition
    u[0, :] = np.sin(np.pi * x)
    
    # Source term function
    def source(x, t):
        return (-np.pi**2 * nu * np.exp(-t) * np.sin(np.pi * x) + 
                np.exp(-t) * np.sin(np.pi * x) - 
                np.pi * np.exp(-2*t) * np.sin(np.pi * x) * np.cos(np.pi * x))
    
    # Time stepping
    for n in range(0, Nt-1):
        # Boundary conditions
        u[n+1, 0] = 0
        u[n+1, -1] = 0
        
        # Interior points
        for i in range(1, Nx-1):
            # Nonlinear convection term (upwind)
            if u[n, i] >= 0:
                conv = u[n, i] * (u[n, i] - u[n, i-1]) / dx
            else:
                conv = u[n, i] * (u[n, i+1] - u[n, i]) / dx
            
            # Diffusion term (central difference)
            diff = nu * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / (dx**2)
            
            # Source term
            src = source(x[i], t[n])
            
            # Update solution
            u[n+1, i] = u[n, i] + dt * (-conv + diff + src)
    
    return x, t, u

# Run simulation
x, t, u = solve_burgers_1d()

# Print some results
print("Solution at t = 0:")
print(u[0, ::10])  # Print every 10th point
print("\nSolution at t = T:")
print(u[-1, ::10])  # Print every 10th point

# Basic visualization (if needed)
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, u[0, :], 'b-', label='t = 0')
    plt.plot(x, u[-1, :], 'r--', label='t = T')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('1D Burgers Equation Solution')
    plt.legend()
    plt.grid(True)
    plt.show()
except ImportError:
    print("Matplotlib not available for visualization")