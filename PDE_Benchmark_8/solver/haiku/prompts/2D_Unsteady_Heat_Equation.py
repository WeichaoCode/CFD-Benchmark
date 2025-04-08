import numpy as np

# Problem parameters
Lx, Ly = 1.0, 1.0  # Domain size
alpha = 1.0  # Diffusion coefficient 
Q0 = 200.0  # Source magnitude
sigma = 0.1  # Source width
Nx, Ny = 50, 50  # Reduced grid points
T_bc = 1.0  # Boundary condition value

# Time parameters
t_start = 0.0
t_end = 3.0

# Grid generation
x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Ly, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Finite difference setup
dx = 2*Lx / (Nx-1)
dy = 2*Ly / (Ny-1)

# Compute stable time step
dt = 0.1 * min(dx, dy)**2 / alpha  # Less restrictive stability criterion
nt = int((t_end - t_start) / dt)

# Initial condition
T = T_bc + Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Vectorized solution approach
for n in range(nt):
    # Compute Laplacian using NumPy's roll function for efficiency
    laplacian_x = (np.roll(T, 1, axis=0) - 2*T + np.roll(T, -1, axis=0)) / dx**2
    laplacian_y = (np.roll(T, 1, axis=1) - 2*T + np.roll(T, -1, axis=1)) / dy**2
    
    # Source term
    Q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Update with source term and diffusion
    T += dt * alpha * (laplacian_x + laplacian_y) + dt * Q
    
    # Apply boundary conditions
    T[0,:] = T_bc
    T[-1,:] = T_bc
    T[:,0] = T_bc
    T[:,-1] = T_bc

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/T_2D_Unsteady_Heat_Equation.npy', T)