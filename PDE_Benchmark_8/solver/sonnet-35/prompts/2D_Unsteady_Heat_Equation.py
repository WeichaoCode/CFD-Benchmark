import numpy as np

# Problem parameters
Lx, Ly = 1.0, 1.0
nx, ny = 100, 100
nt = 300
alpha = 0.01  # Reduced diffusion coefficient
Q0 = 200.0
sigma = 0.1
dx = 2*Lx / (nx-1)
dy = 2*Ly / (ny-1)
dt = 3.0 / nt

# Stability condition check
stability_condition = alpha * dt / (dx**2 + dy**2)
print(f"Stability condition: {stability_condition}")

# Grid generation
x = np.linspace(-Lx, Lx, nx)
y = np.linspace(-Ly, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
T = np.ones((ny, nx)) + Q0 * np.exp(-(X**2 + Y**2)/(2*sigma**2))

# Source term
def source(x, y, t):
    return Q0 * np.exp(-(x**2 + y**2)/(2*sigma**2))

# Time-stepping using explicit finite difference method
for n in range(nt):
    T_old = T.copy()
    
    # Internal points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # 2D heat equation discretization 
            T[i,j] = T_old[i,j] + \
                     alpha*dt/dx**2 * (T_old[i,j+1] - 2*T_old[i,j] + T_old[i,j-1]) + \
                     alpha*dt/dy**2 * (T_old[i+1,j] - 2*T_old[i,j] + T_old[i-1,j]) + \
                     dt * source(X[i,j], Y[i,j], n*dt)
    
    # Boundary conditions
    T[0,:] = 1
    T[-1,:] = 1
    T[:,0] = 1
    T[:,-1] = 1

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/T_2D_Unsteady_Heat_Equation.npy', T)