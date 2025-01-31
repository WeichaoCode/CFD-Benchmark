import numpy as np

# Parameters
Lx = 2.0  # Length of domain in x
Ly = 2.0  # Length of domain in y
Nx = 50   # Number of points in x
Ny = 50   # Number of points in y
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
c = 1.0   # Wave speed
T = 2.0   # Final time
dt = 0.001  # Time step
Nt = int(T/dt)  # Number of time steps

# Create mesh
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize solution array
u = np.zeros((Ny, Nx))
u_new = np.zeros((Ny, Nx))

# Set initial condition
for i in range(Ny):
    for j in range(Nx):
        u[i,j] = np.sin(2*np.pi*x[j]*y[i])

# Time stepping
t = 0
while t < T:
    # Update boundary conditions
    # Bottom boundary (y = 0)
    u[0,:] = 0
    # Left boundary (x = 0)
    u[:,0] = 0
    # Right boundary (x = 2)
    u[:,-1] = np.exp(-t)*np.sin(2*np.pi*y)
    # Top boundary (y = 2)
    u[-1,:] = np.exp(-t)*np.sin(2*np.pi*x)

    # Interior points
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            # Source term
            source = (-np.pi*c*x[j]*np.exp(-t)*np.cos(np.pi*x[j]*y[i]) 
                     - np.pi*c*y[i]*np.exp(-t)*np.cos(np.pi*x[j]*y[i]) 
                     + np.exp(-t)*np.sin(np.pi*x[j]*y[i]))
            
            # Forward difference in time, backward difference in space
            u_new[i,j] = (u[i,j] 
                         - c*dt/dx*(u[i,j] - u[i,j-1])
                         - c*dt/dy*(u[i,j] - u[i-1,j])
                         + dt*source)

    # Update solution
    u = u_new.copy()
    t += dt

    # Print progress
    if int(t/dt) % 100 == 0:
        print(f"Time: {t:.3f}")

# Function to compute exact solution
def exact_solution(x, y, t):
    return np.exp(-t)*np.sin(np.pi*x*y)

# Compute error
u_exact = exact_solution(X, Y, T)
error = np.abs(u - u_exact)
max_error = np.max(error)

print(f"\nMaximum error at final time: {max_error}")

# Print sample values
print("\nNumerical solution at selected points:")
print(f"u(1,1,{T}) = {u[Ny//2,Nx//2]:.6f}")
print(f"Exact solution at (1,1,{T}) = {exact_solution(1,1,T):.6f}")