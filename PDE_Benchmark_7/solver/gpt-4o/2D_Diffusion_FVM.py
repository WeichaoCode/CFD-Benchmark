import numpy as np
import matplotlib.pyplot as plt

# Parameters
h = 0.1  # Length of the square duct (m)
mu = 1e-3  # Dynamic viscosity (Pa·s)
dP_dz = -3.2  # Pressure gradient (Pa/m)

# Grid parameters
nx = ny = 80
dx = dy = h / (nx - 1)

# Coefficients
a_E = a_W = mu * dy / dx
a_N = a_S = mu * dx / dy
a_P = a_E + a_W + a_N + a_S
Su = dP_dz * dx * dy

# Initialize the velocity field w(x, y) to zero initially
w = np.zeros((nx, ny))

# Jacobi iteration parameters
tolerance = 1e-6
max_iterations = 10000
residual = np.inf

# Function to perform Jacobi iteration
def jacobi_iteration(w, a_P, a_E, a_W, a_N, a_S, Su, max_iterations, tolerance):
    for iteration in range(max_iterations):
        w_new = np.copy(w)
        
        # Iterate over the internal nodes
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                w_new[i, j] = (a_E * w[i+1, j] + a_W * w[i-1, j] + 
                               a_N * w[i, j+1] + a_S * w[i, j-1] - Su) / a_P
        
        # Compute the residual
        residual = np.linalg.norm(w_new - w, ord=2)
        
        # Update the solution
        w = w_new

        # Check convergence
        if residual < tolerance:
            print(f'Convergence reached after {iteration+1} iterations with residual {residual:.2e}')
            break

    return w

# Solve the system using Jacobi iteration
w = jacobi_iteration(w, a_P, a_E, a_W, a_N, a_S, Su, max_iterations, tolerance)

# Save the result as a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/w_2D_Diffusion_FVM_SA.npy', w)

# Plotting the result
X, Y = np.meshgrid(np.linspace(0, h, nx), np.linspace(0, h, ny))
plt.contourf(X, Y, w.T, levels=50, cmap='jet')
plt.colorbar(label='Velocity (m/s)')
plt.title('Velocity distribution in the square duct')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.grid(False)
plt.show()