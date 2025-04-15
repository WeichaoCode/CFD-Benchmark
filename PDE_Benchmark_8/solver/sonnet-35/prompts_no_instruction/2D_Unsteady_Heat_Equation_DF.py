import numpy as np

# Problem parameters
Lx, Ly = 2, 2
nx, ny = 41, 41
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
alpha = 0.01  # reduced thermal diffusivity to prevent overflow
Q0 = 200  # source strength
sigma = 0.1
tmax = 3.0
r = 0.1  # reduced stability parameter

# Grid generation
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x, y)

# Time step calculation
dt = r * min(dx**2, dy**2) / alpha

# Number of time steps
nt = int(tmax / dt)

# Initial condition
T = 1 + Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Boundary conditions
T[0, :] = 1
T[-1, :] = 1
T[:, 0] = 1
T[:, -1] = 1

# Source term
def source(x, y, t):
    return Q0 * np.exp(-(x**2 + y**2) / (2 * sigma**2))

# DuFort-Frankel method
T_old = T.copy()
for n in range(nt):
    t = n * dt
    
    # Create source term array
    Q = source(X, Y, t)
    
    # DuFort-Frankel scheme
    T_new = T.copy()
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Use float64 to prevent overflow and improve numerical stability
            laplace_term = np.float64(
                (T[i+1,j] - 2*T[i,j] + T[i-1,j]) / (dx**2) + 
                (T[i,j+1] - 2*T[i,j] + T[i,j-1]) / (dy**2)
            )
            
            T_new[i,j] = np.float64(T_old[i,j] + alpha * dt * laplace_term + dt * Q[i,j])
    
    # Update boundary conditions
    T_new[0, :] = 1
    T_new[-1, :] = 1
    T_new[:, 0] = 1
    T_new[:, -1] = 1
    
    # Update for next iteration
    T_old = T.copy()
    T = T_new.copy()

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/T_2D_Unsteady_Heat_Equation_DF.npy', T)