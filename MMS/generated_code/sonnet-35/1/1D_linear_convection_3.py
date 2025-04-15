import numpy as np
import matplotlib.pyplot as plt

def beam_warming_solver():
    # Parameters
    L = 2.0  # Length of domain
    T = 2.0  # Total time
    c = 1.0  # Wave speed
    
    # Grid parameters
    nx = 80  # Number of spatial points
    dx = L / (nx-1)  # Spatial step size
    
    # CFL condition for stability (CFL ≤ 1 for Beam-Warming)
    CFL = 0.8
    dt = CFL * dx / c
    nt = int(T/dt)  # Number of time steps
    
    # Grid points
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt+1)
    
    # Initialize solution array
    u = np.zeros((nt+1, nx))
    
    # Initial condition
    u[0,:] = np.sin(np.pi * x)
    
    # Source term function
    def source(x, t):
        return -np.pi * c * np.exp(-t) * np.cos(np.pi * x) + np.exp(-t) * np.sin(np.pi * x)
    
    # Time stepping using Beam-Warming scheme
    for n in range(nt):
        for i in range(2, nx):
            # Beam-Warming scheme
            u[n+1,i] = u[n,i] - c * dt/(2*dx) * (3*u[n,i] - 4*u[n,i-1] + u[n,i-2]) \
                       + dt * source(x[i], t[n])
        
        # Boundary conditions
        u[n+1,0] = 0
        u[n+1,1] = 0
        u[n+1,-1] = 0
    
    return x, t, u

# Solve the PDE
x, t, u = beam_warming_solver()

# Plot results at specific time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u[0,:], 'b-', label='t = 0')
plt.plot(x, u[int(len(t)/4),:], 'r--', label=f't = T/4')
plt.plot(x, u[int(len(t)/2),:], 'g-.', label=f't = T/2')
plt.plot(x, u[-1,:], 'k:', label=f't = T')

plt.title('1D Linear Convection - Beam-Warming Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.show()

# Print stability information
dx = x[1] - x[0]
dt = t[1] - t[0]
c = 1.0
CFL = c * dt/dx
print(f"CFL number: {CFL:.3f}")
print("Stability condition: CFL ≤ 1")
print("Scheme is stable" if CFL <= 1 else "Scheme is unstable")

