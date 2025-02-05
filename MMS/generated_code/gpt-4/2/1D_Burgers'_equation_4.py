import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
L = 2        # Length of the domain
T = 2        # Time of simulation
nu = 1       # Viscosity
N = 101     # Number of spatial grid points
M = 2501     # Number of temporal grid points
dx = L/(N-1) # Spatial resolution
dt = T/(M-1) # Temporal resolution
c = 1        # Wave speed

# Initialize the solution array
u = np.zeros((M,N))

# Set the initial condition
u[0,:] = np.sin(np.pi*np.linspace(0,L,N))

# Set the boundary conditions
u[:,0] = 0
u[:,-1] = 0

# Finite difference scheme
for n in range(0,M-1):
    for j in range(1,N-1):
        u[n+1,j] = u[n,j] - c*dt/(2*dx)*(u[n,j+1] - u[n,j-1]) + nu*dt/dx**2*(u[n,j+1] - 2*u[n,j] + u[n,j-1])
        
        # Add source terms
        x = j*dx
        t = n*dt
        u[n+1,j] += dt*(-np.pi**2*nu*np.exp(-t)*np.sin(np.pi*x) + np.exp(-t)*np.sin(np.pi*x) - np.pi*np.exp(-2*t)*np.sin(np.pi*x)*np.cos(np.pi*x))

# Plotting
plt.figure(figsize=(10,6))
plt.plot(np.linspace(0,L,N), u[0,:], label='t = 0')
plt.plot(np.linspace(0,L,N), u[M//4,:], label='t = T/4')
plt.plot(np.linspace(0,L,N), u[M//2,:], label='t = T/2')
plt.plot(np.linspace(0,L,N), u[-1,:], label='t = T')
plt.title('1D Burgers Equation - Forward Time Central Space Scheme')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()